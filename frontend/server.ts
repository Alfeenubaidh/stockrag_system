import express from "express";
import { createServer as createViteServer } from "vite";
import path from "path";
import dotenv from "dotenv";

dotenv.config();

const FASTAPI_URL = process.env.FASTAPI_URL || "http://localhost:8001";

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(express.json());

  app.post("/api/query", async (req, res) => {
    try {
      const { question, ticker, top_k } = req.body;
      const fastapiRes = await fetch(`${FASTAPI_URL}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, ticker, top_k }),
      });
      const data = await fastapiRes.json();
      res.status(fastapiRes.status).json(data);
    } catch (err) {
      res.status(502).json({ error: "FastAPI unreachable" });
    }
  });

  app.get("/api/documents", async (req, res) => {
    try {
      const upstream = await fetch(`${FASTAPI_URL}/documents`);
      const data = await upstream.json();
      res.status(upstream.status).json(data);
    } catch (err) {
      console.error("Documents proxy error:", err);
      res.status(502).json({ error: "Backend unavailable" });
    }
  });

  app.post("/api/query/stream", async (req, res) => {
    try {
      const { question, ticker, top_k } = req.body;
      const fastapiRes = await fetch(`${FASTAPI_URL}/query/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, ticker, top_k }),
      });

      res.setHeader("Content-Type", "text/event-stream");
      res.setHeader("Cache-Control", "no-cache");
      res.setHeader("Connection", "keep-alive");

      if (!fastapiRes.body) {
        res.end();
        return;
      }

      const reader = fastapiRes.body.getReader();
      const pump = async () => {
        const { done, value } = await reader.read();
        if (done) { res.end(); return; }
        res.write(value);
        await pump();
      };
      await pump();
    } catch (err) {
      res.status(502).json({ error: "FastAPI unreachable" });
    }
  });

  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer();
