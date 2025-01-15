import { Configuration, OpenAIApi } from 'openai'
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter'
import { Document } from 'langchain/document'
import { ConversationChain } from 'langchain/chains'
import { PromptTemplate } from 'langchain/prompts'
import { ChatOpenAI } from 'langchain/chat_models/openai'
import { Pool } from 'pg'

interface VectorClientOptions {
  apiKey: string
  dbUrl: string
  model?: string
  template: string
  temperature?: number
  chunkSize?: number
  chunkOverlap?: number
}

interface QueryEmbeddingsOptions {
  embeddings: number[]
  dbFunction: string
  threshold: number
  count: number
}

interface IngestDataOptions {
  data: string
  dbTable: string
}

interface Chunk {
  header: string
  content: string
  prevHeader: string
  nextHeader: string
  chunkId: number
}

interface Metadata {
  header: string
  prevHeader: string
  nextHeader: string
  chunkId: number
  totalChunks: number
  context: string
}

interface Embedding {
  context: string[]
  metadata: Metadata[]
}

export class VectorClient {
  private client: OpenAIApi
  private chatClient: ChatOpenAI
  private dbClient: Pool
  private template: string
  private chunkSize: number
  private chunkOverlap: number

  constructor(options: VectorClientOptions) {
    const {
      apiKey,
      model,
      dbUrl,
      template,
      temperature,
      chunkSize,
      chunkOverlap
    } = options
    this.template = template
    this.chunkSize = chunkSize !== undefined ? chunkSize : 1000
    this.chunkOverlap = chunkOverlap !== undefined ? chunkOverlap : 200
    this.client = null!
    this.chatClient = null!
    this.dbClient = null!

    this.initDb(dbUrl)
    this.initClient(apiKey, model, temperature)
  }

  private initDb(connectionString: string): void {
    try {
      this.dbClient = new Pool({
        connectionString
      })
    } catch (error) {
      throw new Error(
        '[vector-ai] Could not initialize db. Did you provide a valid `dbUrl`?'
      )
    }
  }

  private initClient(
    apiKey: string,
    model?: string,
    temperature?: number
  ): void {
    try {
      this.client = new OpenAIApi(new Configuration({ apiKey }))
      this.chatClient = new ChatOpenAI({
        openAIApiKey: apiKey,
        modelName: model ?? 'gpt-3.5-turbo', // or 'gpt-4'
        temperature: temperature ?? 0
      })
    } catch (error) {
      throw new Error(
        '[vector-ai] Could not initialize client. Did you provide a valid `apiKey` and `model`?'
      )
    }
  }

  private chatTemplate = (): PromptTemplate => {
    const prompt = new PromptTemplate({
      template: this.template,
      inputVariables: ['input']
    })
    return prompt
  }

  /**
   * Ingests data into a specified database table.
   *
   * The flow is as follows:
   * 1. Preprocess the MDX data to split it based on headings.
   * 2. Validate that the first chunk has the required context heading.
   * 3. Calculate the total number of chunks after considering possible sub-chunking of larger chunks.
   * 4. Handle each chunk, further splitting if necessary and inserting into the database.
   *
   * @param {IngestDataOptions} options - Configuration for ingestion, including the data and target database table.
   *
   * @returns A Promise that resolves once all data has been ingested.
   *
   * @example
   * await vectorClient.ingestData({data: "## Title\nThis is some example MDX content.", dbTable: "myTable"});
   *
   * @throws Throws an error if the first chunk is not prefixed with a '## Context' heading, or if there are issues during the ingestion process.
   */
  public async ingestData(options: IngestDataOptions): Promise<void> {
    const { data, dbTable } = options

    const initialChunks = this.preprocessMDX(data)
    this.validateContext(initialChunks[0])

    const totalChunks = await this.calculateTotalChunks(initialChunks)

    for (const chunk of initialChunks) {
      await this.handleChunk(chunk, totalChunks, dbTable)
    }
  }

  /**
   * Validates the context of the first chunk. The first chunk of an MDX document should be prefixed with '## Context'.
   *
   * @param {Chunk} firstChunk - The first chunk from the preprocessed MDX content.
   *
   * @throws Throws an error if the first chunk is not prefixed with '## Context'.
   */
  private validateContext(firstChunk: Chunk): void {
    const contextHeading = firstChunk?.header
    const contextContent = firstChunk?.content

    if (!contextHeading.startsWith('## Context')) {
      throw new Error(
        `[vector-ai] Ingestion Error: a document is not prefixed with a \`## Context\` section as the first section.\n\nHeading: ${contextHeading}\nContent:${contextContent}`
      )
    }
  }

  /**
   * Calculates the total number of chunks expected after considering the division of large chunks into smaller sub-chunks.
   *
   * @param {Chunk[]} initialChunks - An array of chunks from the preprocessed MDX content.
   *
   * @returns {number} The total number of chunks.
   */
  private async calculateTotalChunks(initialChunks: Chunk[]): Promise<number> {
    let extraChunks = 0

    for (const chunk of initialChunks) {
      if (chunk.content.length > this.chunkSize) {
        const textSplitter = this.getTextSplitter()

        const subDocs = await textSplitter.splitDocuments([
          new Document({ pageContent: chunk.content })
        ])

        extraChunks += subDocs.length - 1
      }
    }

    return initialChunks.length + extraChunks
  }

  /**
   * Creates and returns a text splitter instance for dividing large chunks of text.
   *
   * @returns {RecursiveCharacterTextSplitter} An instance of RecursiveCharacterTextSplitter with the configured chunk size and overlap.
   */
  private getTextSplitter(): RecursiveCharacterTextSplitter {
    return new RecursiveCharacterTextSplitter({
      chunkSize: this.chunkSize,
      chunkOverlap: this.chunkOverlap
    })
  }

  /**
   * Handles the processing of a chunk. This involves:
   * 1. Checking if the chunk needs further division based on its size.
   * 2. Assigning metadata to the chunk.
   * 3. Inserting the chunk (and its potential sub-chunks) into the database.
   *
   * @param {Chunk} chunk - The chunk to be handled.
   * @param {number} totalChunks - The total number of chunks calculated.
   * @param {string} dbTable - The name of the target database table.
   *
   * @returns A Promise that resolves once the chunk (and its sub-chunks if any) have been inserted into the database.
   */
  private async handleChunk(
    chunk: Chunk,
    totalChunks: number,
    dbTable: string
  ): Promise<void> {
    let contentToIngest = chunk.content
    const metadata: Metadata = {
      header: chunk.header,
      prevHeader: chunk.prevHeader,
      nextHeader: chunk.nextHeader,
      chunkId: chunk.chunkId,
      totalChunks: totalChunks,
      context: chunk.content
    }

    if (contentToIngest.length > this.chunkSize) {
      const textSplitter = this.getTextSplitter()

      const subDocs = await textSplitter.splitDocuments([
        new Document({ pageContent: contentToIngest })
      ])

      for (const subDoc of subDocs) {
        contentToIngest = subDoc.pageContent.toString().replace(/\n/g, ' ')
        await this.insertIntoDatabase(contentToIngest, metadata, dbTable)
      }
    } else {
      await this.insertIntoDatabase(contentToIngest, metadata, dbTable)
    }
  }

  /**
   * This helper function preprocesses an MDX string by splitting it into logical chunks based on its headings.
   * Each chunk is returned with its associated header and optional adjacent headers (if available).
   *
   * @param {string} mdx - The raw MDX string to be processed.
   * @returns {Chunk[]} - An array of chunks with associated metadata.
   */
  private preprocessMDX(mdx: string): Chunk[] {
    const sections = mdx.split(/(#{1,6} .+\n)/).filter(Boolean)
    const chunks: Chunk[] = []

    for (let i = 0; i < sections.length; i += 2) {
      const header = sections[i].trim()
      const content = sections[i + 1].trim()

      const chunk: Chunk = {
        header,
        content,
        prevHeader: i - 1 >= 0 ? sections[i - 1].trim() : '',
        nextHeader: i + 2 < sections.length ? sections[i + 2].trim() : '',
        chunkId: i / 2 + 1 // +1 to start the IDs from 1 instead of 0
      }

      chunks.push(chunk)
    }

    return chunks
  }

  /**
   * This helper function is responsible for inserting a chunk of content, its embedding, and associated metadata into the database.
   * It first creates an embedding for the content and then performs the insertion.
   *
   * @param {string} content - The chunk of content to be inserted.
   * @param {any} metadata - The metadata associated with the chunk.
   * @param {string} dbTable - The name of the database table where the data will be inserted.
   *
   * @returns A Promise that resolves when the data has been inserted.
   *
   * @throws This method can throw errors if there's an issue with creating embeddings or interacting with the database.
   */
  private async insertIntoDatabase(
    content: string,
    metadata: Metadata,
    dbTable: string
  ): Promise<void> {
    const embeddingContent = `HEADER: ${metadata.header.replace(
      '## ',
      ''
    )} | CONTENT: ${content}`
    const embeddingResponse = await this.createEmbeddings(embeddingContent)
    const embeddings = embeddingResponse.join(',')

    try {
      await this.dbClient.query(
        `INSERT INTO ${dbTable} (content, embedding, metadata) VALUES ($1, ARRAY[${embeddings}]::vector, $2)`,
        [embeddingContent, JSON.stringify(metadata)]
      )
    } catch (error) {
      throw new Error(`Error inserting data into database: ${error}`)
    }
  }

  /**
   * Retries the provided asynchronous function using an exponential backoff strategy.
   *
   * This function is useful when calling external APIs or services that might have temporary
   * rate limits or sporadic failures. Instead of failing immediately, it retries the operation
   * multiple times, waiting longer between each attempt.
   *
   * @param func The asynchronous function to retry.
   * @param maxRetries The maximum number of times to retry the function. Default is 5.
   * @param initialDelayMs The initial delay duration in milliseconds before the first retry.
   *                       Each subsequent retry doubles the delay time. Default is 500ms.
   * @returns The resolved value of the provided function.
   * @throws An error if the function fails after the specified number of retries.
   *
   * @example
   * const data = await retryWithExponentialBackoff(() => externalApi.getData(), 3, 1000);
   */
  private async retryWithExponentialBackoff<T>(
    func: () => Promise<T>,
    maxRetries: number = 5,
    initialDelayMs: number = 500
  ): Promise<T> {
    for (let i = 0; i < maxRetries; i++) {
      try {
        return await func()
      } catch (error) {
        if (i === maxRetries - 1) {
          throw new Error(`Max retries reached: ${error}`)
        }

        // Calculate delay with exponential backoff
        const delayMs = initialDelayMs * Math.pow(2, i)
        await this.sleep(delayMs)
      }
    }
    throw new Error('Retry function reached an unexpected state.')
  }

  /**
   * Sleeps for the specified duration.
   *
   * This function introduces a pause in the execution of asynchronous code,
   * and is typically used to introduce delays between retries or to throttle function calls.
   *
   * @param ms The duration to sleep in milliseconds.
   * @returns A promise that resolves after the specified duration.
   *
   * @example
   * await sleep(1000); // Pauses the execution for 1 second
   */
  private sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms))
  }

  /**
   * Query the model with the question and embedding response context to form answer.
   *
   * @param {string} question - The question for which to get an answer.
   * @returns {Promise<number[]>} A Promise that resolves to an array of numbers representing the embeddings.
   */
  public async createEmbeddings(question: string): Promise<number[]> {
    const getEmbeddings = async (): Promise<number[]> => {
      const embeddingResponse = await this.client.createEmbedding({
        model: 'text-embedding-ada-002',
        input: question
      })
      const [{ embedding }] = embeddingResponse.data.data
      return embedding
    }

    // Note: sometimes these requests get throttled so we have a retry with backoff logic
    // to catch and retry in those scenarios
    return await this.retryWithExponentialBackoff(getEmbeddings)
  }

  /**
   * Query embeddings from the database and return the text form.
   *
   * @param {QueryEmbeddingsOptions} options - The options for the query.
   * @returns {Promise<string>} A Promise that resolves to a formatted string of content separated by '---'.
   */
  public async queryEmbeddings(
    options: QueryEmbeddingsOptions
  ): Promise<Embedding> {
    const { embeddings, dbFunction, threshold, count } = options
    let details: any[] = []
    try {
      const res = await this.dbClient.query(
        `SELECT * FROM ${dbFunction}(ARRAY[${embeddings}]::vector, ${threshold}, ${count})`
      )
      details = res.rows
    } catch (err: any) {
      console.error('Error connecting to database', err.stack)
      throw err
    }

    const context: string[] = details.map((detail) => detail.content.trim())
    const metadata: Metadata[] = details.map((detail) => detail.metadata)

    return { context, metadata }
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
    embedding: Embedding
  ): Promise<string | undefined> {
    const chain = new ConversationChain({
      llm: this.chatClient,
      prompt: this.chatTemplate()
    })

    // Map unique contexts to their associated chunk indices
    // Note, this logic exists to join together chunks that have the same context so as to reduce the token
    // input provided to OpenAI.
    const contextMap: Map<string, number[]> = new Map()

    embedding.metadata.forEach((metadata, index) => {
      if (contextMap.has(metadata.context)) {
        contextMap.get(metadata.context)?.push(index)
      } else {
        contextMap.set(metadata.context, [index])
      }
    })

    const context = [...contextMap.entries()]
      .map(([contextValue, indices], mapIndex) => {
        // Extract content for these indices
        const contents = indices
          .map((index) => embedding.context[index])
          .join('\n')
        return `
        ## Chunk ${mapIndex + 1}

        ### Context
        ${contextValue}

        ### Content
        ${contents}
        ---
      `.trim()
      })
      .join('')

    const template = `
    # Question: {question}
    # Context: {context}
  `.trim()

    const promptA = new PromptTemplate({
      template,
      inputVariables: ['question', 'context']
    })

    const input = await promptA.format({ question, context })

    try {
      const response = await this.retryWithExponentialBackoff(
        () => chain.call({ input }),
        2,
        1000
      )
      return response.response
    } catch (error) {
      console.error('Failed to get an answer after multiple retries:', error)
      // Handle error gracefully or provide a fallback for the user.
      return undefined
    }
  }
}
