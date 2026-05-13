import mongoose from "mongoose";

const connectDB = async (): Promise<boolean> => {
  try {
    const mongoURI = process.env.MONGODB_URI;

    if (!mongoURI) {
      throw new Error("MONGODB_URI is not defined in environment variables");
    }

    const conn = await mongoose.connect(mongoURI);
    console.log(`MongoDB Connected: ${conn.connection.host}`);
    console.log(`MongoDB Database Name: ${conn.connection.name}`);
    return true;
  } catch (error) {
    if (error instanceof Error) {
      console.warn(`MongoDB unavailable: ${error.message}`);
    } else {
      console.warn(`MongoDB unavailable: ${error}`);
    }
    console.warn("Starting server without database-backed features.");
    return false;
  }
};

export default connectDB;
