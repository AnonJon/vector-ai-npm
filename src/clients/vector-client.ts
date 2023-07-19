import { Configuration, OpenAIApi } from "openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { Document } from "langchain/document";
import { ConversationChain } from "langchain/chains";
import { PromptTemplate } from "langchain/prompts";
import { ChatOpenAI } from "langchain/chat_models/openai";
import { Pool } from "pg";

interface VectorClientOptions {
  apiKey: string;
  model: string;
  dbUrl: string;
}

interface QueryEmbeddingsOptions {
  embeddings: number[];
  dbFunction: string;
  threshold: number;
  count: number;
}

interface IngestDataOptions {
  data: string;
  dbTable: string;
  chunkSize?: number;
  chunkOverlap?: number;
}

export class VectorClient {
  private client: OpenAIApi;
  private dbClient: Pool;
  private model: string;
  private chatClient: ChatOpenAI;

  constructor(options: VectorClientOptions) {
    const { apiKey, model, dbUrl } = options;
    this.dbClient = new Pool({
      connectionString: dbUrl,
    });
    this.model = model;
    this.client = new OpenAIApi(new Configuration({ apiKey }));
    this.chatClient = new ChatOpenAI({
      temperature: 0,
      openAIApiKey: apiKey,
      modelName: this.model, // gpt-3.5-turbo, | gpt-4
    });
  }

  private chatTemplate = (): PromptTemplate => {
    const template = `You are a enthusiastic chatbot that loves to answer questions about the StakeDotLink platform. 
    You are talking to a user who is asking you a question about the platform and have some additional context to help answer the question.
     {input}`;
    const prompt = new PromptTemplate({
      template,
      inputVariables: ["input"],
    });
    return prompt;
  };

  /**
   * Query the model with the question and embedding response context to form answer.
   *
   * @param {string} question - The question for which to get an answer.
   * @returns {Promise<number[]>} A Promise that resolves to an array of numbers representing the embeddings.
   */
  public async createEmbeddings(question: string): Promise<number[]> {
    const embeddingResponse = await this.client.createEmbedding({
      model: "text-embedding-ada-002",
      input: question,
    });
    const [{ embedding }] = embeddingResponse.data.data;
    return embedding;
  }

  /**
   * Query embeddings from the database and return the text form.
   *
   * @param {QueryEmbeddingsOptions} options - The options for the query.
   * @returns {Promise<string>} A Promise that resolves to a formatted string of content separated by '---'.
   */
  public async queryEmbeddings(
    options: QueryEmbeddingsOptions
  ): Promise<string> {
    const { embeddings, dbFunction, threshold, count } = options;
    let details: any[] = [];
    try {
      const res = await this.dbClient.query(
        `SELECT * FROM ${dbFunction}(ARRAY[${embeddings}]::vector, ${threshold}, ${count})`
      );
      details = res.rows;
    } catch (err: any) {
      console.error("Error connecting to database", err.stack);
      throw err;
    }

    const context = details
      .map((detail) => `${detail.content.trim()}\n---\n`)
      .join("");

    return context;
  }

  /**
   * Query the model with the question and embedding response context to form answer.
   *
   * @param {string} question - The question for which to get an answer.
   * @param {string} context - The context within which the question should be answered.
   * @returns {Promise<string | undefined>} A Promise that resolves to the text of the answer or undefined if no answer could be generated.
   */
  public async getAnswer(
    question: string,
    context: string
  ): Promise<string | undefined> {
    const chain = new ConversationChain({
      llm: this.chatClient,
      prompt: this.chatTemplate(),
    });
    const template = `Question: {question}
     
    Context: {context}`;
    const promptA = new PromptTemplate({
      template,
      inputVariables: ["question", "context"],
    });
    const input = await promptA.format({ question, context });
    const response = await chain.call({
      input,
    });

    return response.response;
  }

  /**
   * This method ingests data into a specified database table. The data is split into chunks and for each chunk,
   * an embedding is created. The chunk and its corresponding embedding are then inserted into the database.
   *
   * @param {IngestDataOptions} options - The options for the ingestion. chunkSize and chunkOverlap are optional and default to 1000 and 200 respectively.
   *
   * @returns A Promise that resolves when all data has been ingested.
   *
   * @example
   * await vectorClient.ingestData({data: "This is some example data.", dbTable: "myTable"});
   *
   * @throws This method can throw errors if there's an issue with splitting documents, creating embeddings,
   * or interacting with the database.
   */
  public async ingestData(options: IngestDataOptions): Promise<void> {
    const { data, dbTable, chunkSize = 1000, chunkOverlap = 200 } = options;
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize,
      chunkOverlap,
    });
    const docs = await textSplitter.splitDocuments([
      new Document({ pageContent: data }),
    ]);

    for (const doc of docs) {
      const input = doc.pageContent.toString().replace(/\n/g, " ");

      const embeddingResponse = await this.createEmbeddings(input);
      const embeddings = embeddingResponse.join(",");

      try {
        await this.dbClient.query(
          `INSERT INTO ${dbTable} (content, embedding) VALUES ($1, ARRAY[${embeddings}]::vector)`,
          [doc.pageContent.toString()]
        );
      } catch (error) {
        throw new Error(`Error inserting data into database: ${error}`);
      }
    }
  }
}
