
import { useState } from "react";
import { detectText } from "../services/textDetectionApi";
import PredictionExplanation from "../components/PredictionExplanation";
import "./TextDetection.css";

const Icon = ({ name, size = 20 }) => {
  const paths = {
    sparkles: (
      <>
        <path d="m12 3 1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8L12 3Z" />
        <path d="m19 17 .7 2.3L22 20l-2.3.7L19 23l-.7-2.3L16 20l2.3-.7L19 17Z" />
      </>
    ),
    file: (
      <>
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z" />
        <path d="M14 2v6h6M8 13h8M8 17h6" />
      </>
    ),
    chart: (
      <>
        <path d="M3 3v18h18" />
        <path d="m7 16 4-5 4 3 5-8" />
      </>
    ),
    shield: (
      <>
        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10Z" />
        <path d="m9 12 2 2 4-4" />
      </>
    ),
    arrow: <path d="M5 12h14m-6-6 6 6-6 6" />,
    refresh: (
      <>
        <path d="M20 11a8 8 0 0 0-14-5L4 8" />
        <path d="M4 4v4h4" />
        <path d="M4 13a8 8 0 0 0 14 5l2-2" />
        <path d="M20 20v-4h-4" />
      </>
    ),
    check: <path d="m5 12 4 4L19 6" />,
    alert: (
      <>
        <circle cx="12" cy="12" r="10" />
        <path d="M12 7v5m0 4h.01" />
      </>
    ),
    menu: <path d="M4 7h16M4 12h16M4 17h16" />,
    brain: (
      <>
        <path d="M12 18V5a3 3 0 0 0-5.8-1A4 4 0 0 0 4 11a4 4 0 0 0 1 7 4 4 0 0 0 7 0Z" />
        <path d="M12 18V5a3 3 0 0 1 5.8-1A4 4 0 0 1 20 11a4 4 0 0 1-1 7 4 4 0 0 1-7 0Z" />
      </>
    ),
  };

  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {paths[name] || paths.file}
    </svg>
  );
};

function ResultCard({ label, value, subtext }) {
  return (
    <div className="result-card">
      <span className="result-card-label">{label}</span>
      <strong className="result-card-value">{value}</strong>
      {subtext && <small>{subtext}</small>}
    </div>
  );
}

function getShortExplanation(prediction, truncated) {
  let explanation;

  switch (prediction) {
    case "AI_LIKE":
      explanation =
        "The model found patterns in the analyzed text that " +
        "increased its AI-class score. The highlighted words " +
        "show which parts influenced this result. This does " +
        "not prove that AI generated the text.";
      break;

    case "HUMAN_LIKE":
      explanation =
        "The model found patterns in the analyzed text that " +
        "reduced its AI-class score. The highlighted words " +
        "show which parts influenced this result. This does " +
        "not prove that a human wrote the text.";
      break;

    case "UNCERTAIN":
      explanation =
        "The model's score falls between the AI-like and " +
        "human-like classification thresholds. It cannot " +
        "provide a clear classification for this input.";
      break;

    case "INSUFFICIENT_TEXT":
      explanation =
        "The submitted text is too short for analysis. " +
        "Please provide at least 50 words.";
      break;

    default:
      explanation =
        "The model returned a result that requires further " +
        "interpretation.";
  }

  if (truncated) {
    explanation +=
      " Only the beginning of the text was analyzed " +
      "because of the model's 128-token limit.";
  }

  return explanation;
}

export default function TextDetection() {
  const [text, setText] = useState("");
  const [analyzedText, setAnalyzedText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const wordCount = text.trim()
    ? text.trim().split(/\s+/).length
    : 0;

  const characterCount = text.length;
  const canAnalyze = wordCount >= 50 && !loading;

  const aiScore =
    typeof result?.ai_score === "number"
      ? Math.max(0, Math.min(1, result.ai_score))
      : null;

  function getPredictionLabel(prediction) {
    const labels = {
      AI_LIKE: "AI-like writing",
      HUMAN_LIKE: "Human-like writing",
      UNCERTAIN: "Uncertain result",
      INSUFFICIENT_TEXT: "Insufficient text",
    };

    return labels[prediction] || prediction || "Unknown";
  }

  function getPredictionStyle(prediction) {
    if (prediction === "AI_LIKE") return "ai";
    if (prediction === "HUMAN_LIKE") return "human";
    return "uncertain";
  }

  async function handleAnalyze() {
    if (!canAnalyze) return;

    const submittedText = text.trim();

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await detectText(submittedText);

      setAnalyzedText(submittedText);
      setResult(response);
    } catch (err) {
      setError(
        err.message ||
          "Something went wrong while analyzing your text."
      );
    } finally {
      setLoading(false);
    }
  }

  function handleClear() {
    setText("");
    setAnalyzedText("");
    setResult(null);
    setError("");
  }

  function handleTextChange(event) {
    setText(event.target.value);
    setResult(null);
    setError("");
  }

  return (
    <div className="dashboard-layout">
      {sidebarOpen && (
        <div
          className="sidebar-overlay"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* SIDEBAR */}
      <aside className={`sidebar ${sidebarOpen ? "open" : ""}`}>
        <div className="brand">
          <div className="brand-icon">
            <Icon name="sparkles" size={24} />
          </div>

          <div>
            <strong>Verity AI</strong>
            <span>Content Intelligence</span>
          </div>
        </div>

        <div className="sidebar-section-label">WORKSPACE</div>

        <nav className="sidebar-nav">
          <div className="nav-item active">
            <Icon name="file" />
            <span>Text Detection</span>
          </div>

          <div className="nav-item disabled">
            <Icon name="chart" />
            <span>Analytics</span>
            <small>Soon</small>
          </div>

          <div className="nav-item disabled">
            <Icon name="shield" />
            <span>Provenance</span>
            <small>Soon</small>
          </div>
        </nav>

        <div className="sidebar-bottom">
          <div className="sidebar-info">
            <Icon name="brain" />
            <strong>Explainable AI</strong>
            <p>
              Explore model predictions and inspect words
              that influence the classification.
            </p>
          </div>

          <span className="sidebar-version">
            Version 0.2 · Research MVP
          </span>
        </div>
      </aside>

      {/* MAIN CONTENT */}
      <div className="main-content">
        <header className="topbar">
          <div className="topbar-left">
            <button
              type="button"
              className="mobile-menu"
              onClick={() => setSidebarOpen(!sidebarOpen)}
              aria-label="Toggle navigation"
            >
              <Icon name="menu" />
            </button>

            <span>Workspace</span>
            <span className="breadcrumb-separator">/</span>
            <strong>Text Detection</strong>
          </div>

          <div className="topbar-status">
            <span className="status-dot" />
            Experimental Model
          </div>
        </header>

        <main className="page-content">
          {/* HERO */}
          <section className="hero">
            <div className="hero-decoration hero-decoration-one" />
            <div className="hero-decoration hero-decoration-two" />

            <div className="hero-content">
              <div className="hero-pill">
                <Icon name="sparkles" size={15} />
                EXPLAINABLE AI DETECTION
              </div>

              <h1>
                Discover the story
                <br />
                <span>behind your text.</span>
              </h1>

              <p>
                Analyze writing patterns with deep learning,
                explore the prediction, and understand which
                words influence the model's decision.
              </p>

              <div className="hero-features">
                <span>
                  <Icon name="check" size={15} />
                  AI content analysis
                </span>

                <span>
                  <Icon name="check" size={15} />
                  Explainable predictions
                </span>
              </div>
            </div>

            <div className="hero-art" aria-hidden="true">
              <div className="art-orbit art-orbit-one" />
              <div className="art-orbit art-orbit-two" />

              <div className="art-center">
                <Icon name="brain" size={44} />
              </div>
            </div>
          </section>

          {/* INPUT */}
          <div className="section-heading">
            <div>
              <span className="section-eyebrow">
                DETECTION STUDIO
              </span>

              <h2>Analyze your content</h2>

              <p>
                Paste your text below to begin a new analysis.
              </p>
            </div>

            <span className="section-tag">
              <Icon name="shield" size={15} />
              DistilRoBERTa V2
            </span>
          </div>

          <section className="editor-card">
            <div className="editor-header">
              <div className="editor-title">
                <div className="editor-icon">
                  <Icon name="file" />
                </div>

                <div>
                  <h3>Text Input</h3>
                  <span>Minimum 50 words required</span>
                </div>
              </div>

              <button
                type="button"
                className="clear-button"
                onClick={handleClear}
                disabled={loading || (!text && !result)}
              >
                <Icon name="refresh" size={16} />
                Clear
              </button>
            </div>

            <textarea
              className="text-editor"
              value={text}
              onChange={handleTextChange}
              placeholder={
                "Start typing or paste your content here...\n\n" +
                "Your text will be analyzed for patterns " +
                "associated with AI-generated writing."
              }
              maxLength={100000}
              rows={11}
              disabled={loading}
            />

            <div className="editor-footer">
              <div className="editor-stats">
                <span>
                  <strong>{wordCount}</strong> words
                </span>

                <span className="stat-divider" />

                <span>
                  <strong>{characterCount}</strong> characters
                </span>
              </div>

              <button
                type="button"
                className="analyze-button"
                onClick={handleAnalyze}
                disabled={!canAnalyze}
              >
                {loading ? (
                  <>
                    <span className="loading-spinner" />
                    Analyzing & explaining...
                  </>
                ) : (
                  <>
                    <Icon name="sparkles" size={18} />
                    Analyze Content
                    <Icon name="arrow" size={18} />
                  </>
                )}
              </button>
            </div>

            {wordCount > 0 && wordCount < 50 && (
              <div className="word-hint">
                Add {50 - wordCount} more words to enable
                analysis.
              </div>
            )}

            {error && (
              <div className="error-banner" role="alert">
                <Icon name="alert" />
                <span>{error}</span>
              </div>
            )}
          </section>

          {/* RESULTS */}
          {result ? (
            <section className="results-section">
              <div className="section-heading">
                <div>
                  <span className="section-eyebrow">
                    ANALYSIS COMPLETE
                  </span>

                  <h2>Your detection results</h2>

                  <p>
                    Review the classification and explanation.
                  </p>
                </div>
              </div>

              {/* PREDICTION */}
              <div className="prediction-panel">
                <div className="prediction-main">
                  <span className="result-overline">
                    MODEL CLASSIFICATION
                  </span>

                  <div
                    className={`prediction-badge ${getPredictionStyle(
                      result.prediction
                    )}`}
                  >
                    <span className="prediction-indicator" />
                    {getPredictionLabel(result.prediction)}
                  </div>

                  <p>
                    This result describes patterns recognized
                    by the classifier. It does not prove
                    whether a human or AI wrote the text.
                  </p>
                </div>

                <div className="prediction-score">
                  <span>Raw AI-class score</span>

                  <strong>
                    {aiScore === null
                      ? "N/A"
                      : `${(aiScore * 100).toFixed(1)}%`}
                  </strong>

                  <small>Not a calibrated probability</small>
                </div>
              </div>

              {/* SHORT EXPLANATION */}
              <div
                className="explanation-info"
                style={{
                  marginTop: "18px",
                  marginBottom: "20px",
                  padding: "22px",
                  borderRadius: "14px",
                }}
              >
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "10px",
                    marginBottom: "10px",
                  }}
                >
                  <Icon name="brain" size={22} />

                  <h3
                    style={{
                      margin: 0,
                      fontSize: "16px",
                    }}
                  >
                    Why did the model predict this?
                  </h3>
                </div>

                <p
                  style={{
                    margin: 0,
                    fontSize: "13px",
                    lineHeight: 1.9,
                  }}
                >
                  {getShortExplanation(
                    result.prediction,
                    result.truncated
                  )}
                </p>
              </div>

              {/* SCORE DISTRIBUTION */}
              {aiScore !== null && (
                <div className="score-card">
                  <div className="score-card-header">
                    <h3>Model score distribution</h3>
                    <span>Experimental output</span>
                  </div>

                  <div className="score-labels">
                    <span>Human-class score</span>
                    <span>AI-class score</span>
                  </div>

                  <div className="score-bar">
                    <div
                      className="score-bar-fill"
                      style={{
                        width: `${aiScore * 100}%`,
                      }}
                    />
                  </div>

                  <div className="score-values">
                    <strong>
                      {((1 - aiScore) * 100).toFixed(1)}%
                    </strong>

                    <strong>
                      {(aiScore * 100).toFixed(1)}%
                    </strong>
                  </div>
                </div>
              )}

              {/* TEXT STATISTICS */}
              <div className="result-grid">
                <ResultCard
                  label="Word count"
                  value={result.word_count}
                  subtext="Words submitted"
                />

                <ResultCard
                  label="Token count"
                  value={result.token_count ?? "N/A"}
                  subtext="Tokens in submitted text"
                />

                <ResultCard
                  label="Input coverage"
                  value={
                    result.truncated ? "Partial" : "Full"
                  }
                  subtext={
                    result.truncated
                      ? "Limited to 128 model tokens"
                      : "Within the model token limit"
                  }
                />
              </div>

              {result.truncated && (
                <div className="error-banner" role="alert">
                  <Icon name="alert" />

                  <span>
                    The submitted text exceeds the 128-token
                    model limit. The prediction and explanation
                    cover only the beginning of the text.
                  </span>
                </div>
              )}

              {/* DETAILED EXPLAINABLE AI */}
              {result.explanation ? (
                <PredictionExplanation
                  result={result}
                  text={analyzedText}
                />
              ) : (
                <div className="disclaimer-card">
                  <Icon name="alert" size={23} />

                  <div>
                    <strong>
                      Detailed explanation not available
                    </strong>

                    <p>
                      The backend returned a prediction without
                      word-level explanation data. Verify that
                      the updated prediction_service.py is running.
                    </p>
                  </div>
                </div>
              )}

              {/* WARNING */}
              <div className="disclaimer-card">
                <Icon name="shield" size={23} />

                <div>
                  <strong>
                    Important interpretation note
                  </strong>

                  <p>
                    {result.warning ||
                      "AI detection is experimental and cannot prove authorship."}
                    {" "}
                    This model has not yet demonstrated reliable
                    generalization to independent real-world
                    datasets.
                  </p>
                </div>
              </div>
            </section>
          ) : (
            <div className="empty-results">
              <div className="empty-icon">
                <Icon name="brain" size={26} />
              </div>

              <h3>Your analysis will appear here</h3>

              <p>
                Submit at least 50 words to explore the
                prediction, a short explanation, model scores,
                and highlighted influential words.
              </p>
            </div>
          )}

          {/* FOOTER */}
          <footer className="page-footer">
            <span>Verity AI · Research Prototype</span>

            <span>
              React · FastAPI · DistilRoBERTa · Explainable AI
            </span>
          </footer>
        </main>
      </div>
    </div>
  );
}
