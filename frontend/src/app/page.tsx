"use client";

import { useState } from "react";

export default function Home() {
  const [image, setImage] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleImageChange = (file: File | undefined) => {
    if (!file) return;

    setImage(file);
    setPreview(URL.createObjectURL(file));
    setAnswer("");
    setError("");
  };

  const askQuestion = async () => {
    if (!image || !question.trim()) return;

    setLoading(true);
    setAnswer("");
    setError("");

    try {
      const formData = new FormData();

      formData.append("image", image);
      formData.append("question", question);

      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Something went wrong.");
      }

      setAnswer(data.answer);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to connect to VisionQuery API."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-slate-950 text-white">
      <div className="mx-auto flex min-h-screen max-w-5xl flex-col px-6 py-10">
        {/* Header */}
        <header className="mb-10 text-center">
          <div className="mb-3 text-4xl">👁️</div>

          <h1 className="text-4xl font-bold tracking-tight">
            VisionQuery
          </h1>

          <p className="mt-3 text-slate-400">
            Ask questions about your images using AI.
          </p>
        </header>

        {/* Main Card */}
        <section className="rounded-3xl border border-slate-800 bg-slate-900 p-6 shadow-2xl">
          {/* Upload */}
          <div className="mb-6">
            <label className="mb-2 block text-sm font-medium text-slate-300">
              Upload an image
            </label>

            <input
              type="file"
              accept="image/*"
              onChange={(e) => handleImageChange(e.target.files?.[0])}
              className="block w-full cursor-pointer rounded-xl border border-slate-700 bg-slate-950 p-3 text-sm text-slate-300 file:mr-4 file:rounded-lg file:border-0 file:bg-indigo-600 file:px-4 file:py-2 file:text-sm file:font-medium file:text-white hover:file:bg-indigo-500"
            />
          </div>

          {/* Image Preview */}
          {preview && (
            <div className="mb-6 overflow-hidden rounded-2xl border border-slate-800">
              <img
                src={preview}
                alt="Uploaded image"
                className="max-h-[420px] w-full object-contain bg-black"
              />
            </div>
          )}

          {/* Question */}
          <div className="mb-6">
            <label className="mb-2 block text-sm font-medium text-slate-300">
              Ask a question
            </label>

            <input
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  askQuestion();
                }
              }}
              placeholder="e.g. What is in the image?"
              className="w-full rounded-xl border border-slate-700 bg-slate-950 px-4 py-3 text-white outline-none placeholder:text-slate-600 focus:border-indigo-500"
            />
          </div>

          {/* Ask Button */}
          <button
            onClick={askQuestion}
            disabled={!image || !question.trim() || loading}
            className="w-full rounded-xl bg-indigo-600 px-5 py-3 font-semibold transition hover:bg-indigo-500 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {loading ? "VisionQuery is thinking..." : "Ask VisionQuery"}
          </button>

          {/* Error */}
          {error && (
            <div className="mt-6 rounded-xl border border-red-900 bg-red-950/40 p-4 text-sm text-red-300">
              {error}
            </div>
          )}

          {/* Answer */}
          {answer && (
            <div className="mt-6 rounded-2xl border border-indigo-500/30 bg-indigo-950/30 p-5">
              <p className="mb-2 text-sm font-medium text-indigo-300">
                AI Answer
              </p>

              <p className="text-2xl font-semibold">
                {answer}
              </p>
            </div>
          )}
        </section>

        {/* Footer */}
        <footer className="mt-auto pt-8 text-center text-sm text-slate-600">
          Powered by fine-tuned BLIP · VisionQuery
        </footer>
      </div>
    </main>
  );
}