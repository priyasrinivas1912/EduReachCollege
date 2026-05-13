import app from "./app.ts";
import connectDB from "./config/database.config.ts";
import { initializeKnowledgeBase } from "./services/rag.service.ts";

const PORT = process.env.PORT || 5000;

const start = async (): Promise<void> => {
  try {
    const isDatabaseConnected = await connectDB();

    if (isDatabaseConnected) {
      try {
        await initializeKnowledgeBase();
      } catch (error) {
        console.warn("Knowledge base initialization failed. Continuing without AI chat setup.");
        console.warn(error);
      }
    }

    app.listen(PORT, () => {
      console.log("EduReach Server is running!");
      console.log(`URL: http://localhost:${PORT}`);
      console.log(`Node: ${process.version}`);
      console.log("Press Ctrl+C to stop");
    });
  } catch (error) {
    console.error("Failed to start server:", error);
    process.exit(1);
  }
};

start();
