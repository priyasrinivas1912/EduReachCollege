import path from "node:path";
import { fileURLToPath } from "node:url";
import { MongoClient } from "mongodb";
import { createAgent, tool } from "langchain";
import {
  ChatGoogleGenerativeAI,
  GoogleGenerativeAIEmbeddings,
} from "@langchain/google-genai";
import { MongoDBAtlasVectorSearch } from "@langchain/mongodb";
import { TextLoader } from "@langchain/classic/document_loaders/fs/text";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { z } from "zod";

// ---- __dirname for ESM ----
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ---- MongoDB native client ----
let mongoClient: MongoClient | null = null;
let knowledgeTextCache: string | null = null;

const getMongoClient = async (): Promise<MongoClient> => {
  if (!mongoClient) {
    mongoClient = new MongoClient(process.env.MONGODB_URI || "");
    await mongoClient.connect();
  }
  return mongoClient;
};

const getKnowledgeFilePath = () => path.join(__dirname, "../../knowledge-base/edureach-knowledge.txt");

const loadKnowledgeText = async (): Promise<string> => {
  if (knowledgeTextCache) {
    return knowledgeTextCache;
  }

  const loader = new TextLoader(getKnowledgeFilePath());
  const docs = await loader.load();
  knowledgeTextCache = docs.map((doc) => doc.pageContent).join("\n\n");
  return knowledgeTextCache;
};

// ---- Google GenAI Embeddings ----
// gemini-embedding-001 → default 3072 dimensions (FREE, same API key as Gemini chat)
const getEmbeddings = () => {
  if (!process.env.GOOGLE_API_KEY) {
    throw new Error("GOOGLE_API_KEY is not set in .env!");
  }
  return new GoogleGenerativeAIEmbeddings({
    apiKey: process.env.GOOGLE_API_KEY,
    model: "gemini-embedding-001",
  });
};

// ---- Vector Store ----
const getVectorStore = async () => {
  const client = await getMongoClient();
  const collection = client.db("edureach_db").collection("knowledge_docs");

  return new MongoDBAtlasVectorSearch(getEmbeddings(), {
    collection: collection as any,
    indexName: "edureach_vector_index",
    textKey: "text",
    embeddingKey: "embedding",
  });
};

// ============================================
// A) INDEXING — runs ONCE at server startup
// ============================================
export const initializeKnowledgeBase = async (): Promise<void> => {
  const client = await getMongoClient();
  const collection = client.db("edureach_db").collection("knowledge_docs");

  // Check if docs exist WITH valid (non-empty) embeddings
  const docWithEmbedding = await collection.findOne({
    embedding: { $exists: true, $not: { $size: 0 } },
  });

  if (docWithEmbedding) {
    const count = await collection.countDocuments();
    console.log(` Knowledge base ready (${count} chunks with embeddings)`);
    return;
  }

  // If docs exist but embeddings are empty → delete and re-index
  const existingCount = await collection.countDocuments();
  if (existingCount > 0) {
    console.log(` Found ${existingCount} chunks with EMPTY embeddings — deleting & re-indexing...`);
    await collection.deleteMany({});
  }

  console.log(" Indexing knowledge base...");

  // Verify API key FIRST with a test embedding
  const embeddings = getEmbeddings();
  try {
    const testResult = await embeddings.embedQuery("test");
    console.log(` API key OK — embedding dimensions: ${testResult.length}`);
  } catch (error: any) {
    console.error(" Embedding test failed!");
    console.error("   Error:", error.message || error);
    console.error("   Get key from: https://aistudio.google.com/apikey");
    throw error;
  }

  // LOAD
  const filePath = getKnowledgeFilePath();
  const loader = new TextLoader(filePath);
  const docs = await loader.load();
  if (docs.length === 0) {
    throw new Error("No documents found in knowledge base file");
  }
  const totalCharacters = docs.reduce((sum, doc) => sum + doc.pageContent.length, 0);
  console.log(`    Loaded ${totalCharacters} characters`);

  // SPLIT
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });
  const allSplits = await splitter.splitDocuments(docs);
  console.log(`    Split into ${allSplits.length} chunks`);

  // EMBED + STORE
  const vectorStore = new MongoDBAtlasVectorSearch(embeddings, {
    collection: collection as any,
    indexName: "edureach_vector_index",
    textKey: "text",
    embeddingKey: "embedding",
  });

  await vectorStore.addDocuments(allSplits);

  // VERIFY
  const verifyDoc = await collection.findOne({
    embedding: { $exists: true, $not: { $size: 0 } },
  });

  if (verifyDoc && Array.isArray(verifyDoc.embedding) && verifyDoc.embedding.length > 0) {
    console.log(`    ${allSplits.length} chunks stored (${verifyDoc.embedding.length}D embeddings)`);
    console.log(`     IMPORTANT: Create Atlas Vector Search index with numDimensions: ${verifyDoc.embedding.length}`);
  } else {
    await collection.deleteMany({});
    throw new Error(" Embeddings are empty! Google API returned no vectors.");
  }
};

// ============================================
// B) AGENT — runs on every chat query
// ============================================
const createRetrieveTool = (vectorStore: MongoDBAtlasVectorSearch) => {
  return tool(
    async ({ query }: { query: string }) => {
      const retrievedDocs = await vectorStore.similaritySearch(query, 3);
      return retrievedDocs
        .map((doc) => `Source: ${doc.metadata.source}\nContent: ${doc.pageContent}`)
        .join("\n\n");
    },
    {
      name: "retrieve",
      description:
        "Retrieve information from the EduReach College knowledge base. " +
        "Use this for any questions about courses, fees, admissions, mentors, campus, placements.",
      schema: z.object({ query: z.string() }),
    }
  );
};

const fallbackSections = [
  {
    title: "fees",
    keywords: ["fee", "fees", "tuition", "hostel fee", "lab fee", "exam fee", "payment", "installment", "emi"],
    answer:
      "For 2024-2025, B.Tech tuition is Rs 1,50,000 per year. Hostel fee is Rs 80,000, lab fee is Rs 15,000, and exam fee is Rs 5,000. Total is Rs 1,70,000 for day scholars and Rs 2,50,000 for hostellers. Fees can be paid in 2 installments, and education loans plus EMI options are available.",
  },
  {
    title: "btech cse fee",
    keywords: ["b.tech cse", "cse fee", "computer science fee", "btech computer science"],
    answer:
      "For B.Tech CSE, the fee follows the general B.Tech structure: tuition is Rs 1,50,000 per year, lab fee is Rs 15,000, and exam fee is Rs 5,000. That makes Rs 1,70,000 per year for a day scholar, or Rs 2,50,000 per year with hostel.",
  },
  {
    title: "courses",
    keywords: ["course", "courses", "program", "programs", "offer", "offered", "branch", "branches"],
    answer:
      "EduReach offers B.Tech in CSE, ECE, AI and DS, Mechanical, Civil, and IT. It also offers M.Tech in Computer Science, VLSI Design, and Structural Engineering, plus an MBA with Finance, Marketing, HR, and IT specializations.",
  },
  {
    title: "admissions",
    keywords: ["admission", "admissions", "apply", "application", "eligibility", "eamcet", "jee", "gate", "icet"],
    answer:
      "For B.Tech admissions, 70 percent of seats are through TS EAMCET or AP EAMCET counseling and 30 percent are through management quota. Eligibility is 10+2 with PCM and at least 60 percent. Applications open on March 1, management quota closes on July 15, and classes begin on August 1 for the odd semester.",
  },
  {
    title: "placements",
    keywords: ["placement", "placements", "package", "recruiter", "recruiters", "average", "highest"],
    answer:
      "EduReach reports a 92 percent placement rate for 2023-2024. The highest package is Rs 42 LPA, average package is Rs 8.5 LPA, and median package is Rs 6.5 LPA. Top recruiters include Google, Microsoft, Amazon, TCS, Infosys, Wipro, Deloitte, Accenture, Razorpay, and PhonePe.",
  },
  {
    title: "hostel",
    keywords: ["hostel", "hosteller", "mess", "ac room", "non-ac", "security"],
    answer:
      "Hostel is optional. EduReach has separate hostels for boys and girls with AC and non-AC rooms, mess facilities, gym, indoor games, laundry, and 24/7 security with CCTV. Hostel fee is Rs 80,000 per year including mess.",
  },
  {
    title: "campus",
    keywords: ["campus", "library", "lab", "labs", "sports", "club", "clubs", "event", "events", "transport"],
    answer:
      "EduReach has a 25-acre green campus with smart classrooms, advanced labs, a central library with 50,000 plus books, sports facilities, clubs, hostels, and transport from 15 plus routes across Hyderabad.",
  },
];

const getFallbackResponse = async (question: string): Promise<string> => {
  const normalizedQuestion = question.toLowerCase();
  const bestSection = fallbackSections
    .map((section) => ({
      section,
      score: section.keywords.filter((keyword) => normalizedQuestion.includes(keyword)).length,
    }))
    .sort((a, b) => b.score - a.score)[0];

  if (bestSection && bestSection.score > 0) {
    return bestSection.section.answer;
  }

  const knowledgeText = await loadKnowledgeText();
  const lines = knowledgeText.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
  const matchingLine = lines.find((line) => {
    const normalizedLine = line.toLowerCase();
    return normalizedQuestion
      .split(/\W+/)
      .filter((word) => word.length > 3)
      .some((word) => normalizedLine.includes(word));
  });

  if (matchingLine) {
    return `From the EduReach knowledge base: ${matchingLine}`;
  }

  return "I don't have that information right now. Click Talk to Us to speak with a counselor.";
};

export const getRAGResponse = async (question: string): Promise<string> => {
  try {
    const vectorStore = await getVectorStore();
    const retrieve = createRetrieveTool(vectorStore);

    const model = new ChatGoogleGenerativeAI({
      model: "gemini-2.5-flash",
      temperature: 0.7,
    });

    const agent = createAgent({
      model,
      tools: [retrieve],
      systemPrompt:
        "You are EduReach Bot, a helpful AI counselor for EduReach College, Hyderabad. " +
        "ALWAYS use the retrieve tool to search the knowledge base before answering. " +
        "Be concise, friendly, and professional. " +
        "If the information is not found, say: " +
        "'I don't have that information right now. Click Talk to Us to speak with a counselor.'",
    });

    const result = await agent.invoke({
      messages: [{ role: "user", content: question }],
    });

    const messages = result.messages;
    const lastMessage = messages[messages.length - 1];

    if (!lastMessage) {
      return "I couldn't generate a response. Please try again.";
    }

    return typeof lastMessage.content === "string"
      ? lastMessage.content
      : JSON.stringify(lastMessage.content);
  } catch (error) {
    console.error(" RAG Agent Error:", error);
    return getFallbackResponse(question);
  }
};
