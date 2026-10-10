
import "./PredictionExplanation.css";

export default function PredictionExplanation({ result, text }) {
  const explanation = result?.explanation;

  if (!explanation) return null;

  const highlights = explanation.highlights || [];

  // The backend returns character offsets into the trimmed text.
  const analyzedText = text.trim();

  const ordered = [...highlights].sort(
    (a, b) => a.start - b.start
  );

  function renderHighlightedText() {
    const parts = [];
    let cursor = 0;

    ordered.forEach((item, index) => {
      if (
        item.start < cursor ||
        item.end > analyzedText.length
      ) {
        return;
      }

      parts.push(
        analyzedText.slice(cursor, item.start)
      );

      parts.push(
        <mark
          key={index}
          className={
            item.direction === "AI"
              ? "highlight-ai"
              : item.direction === "HUMAN"
              ? "highlight-human"
              : "highlight-neutral"
          }
          title={`AI-score impact: ${item.impact}`}
        >
          {analyzedText.slice(item.start, item.end)}
        </mark>
      );

      cursor = item.end;
    });

    parts.push(analyzedText.slice(cursor));

    return parts;
  }

  const aiWords = highlights.filter(
    (item) => item.impact > 0
  );

  const humanWords = highlights.filter(
    (item) => item.impact < 0
  );

  return (
    <section className="explanation-panel">
      <div className="explanation-heading">
        <span className="explanation-eyebrow">
          EXPLAINABLE AI
        </span>
        <h2>Why did the model predict this?</h2>
        <p>{explanation.summary}</p>
      </div>

      <div className="explanation-info">
        <strong>How to read this explanation</strong>
        <p>
          Each highlighted word was masked to measure
          how much the AI-class score changed.
          This identifies influential input words,
          not reliable evidence of who wrote the text.
        </p>
      </div>

      <div className="explanation-legend">
        <span>
          <i className="legend-dot ai-dot" />
          Supports AI-class score
        </span>
        <span>
          <i className="legend-dot human-dot" />
          Supports human-class score
        </span>
      </div>

      <div className="highlighted-content">
        {renderHighlightedText()}
      </div>

      <div className="influence-grid">
        <div className="influence-card">
          <h3>AI-class supporting words</h3>

          {aiWords.length ? (
            aiWords.map((item, index) => (
              <div className="influence-row" key={index}>
                <span>{item.word}</span>
                <strong>+{item.impact.toFixed(4)}</strong>
              </div>
            ))
          ) : (
            <p>No positive influences in the top results.</p>
          )}
        </div>

        <div className="influence-card">
          <h3>Human-class supporting words</h3>

          {humanWords.length ? (
            humanWords.map((item, index) => (
              <div className="influence-row" key={index}>
                <span>{item.word}</span>
                <strong>{item.impact.toFixed(4)}</strong>
              </div>
            ))
          ) : (
            <p>No negative influences in the top results.</p>
          )}
        </div>
      </div>

      <p className="explanation-footer">
        Method: Word-level token occlusion ·
        Evaluated words: {explanation.evaluated_words ?? 0}
        {result.truncated
          ? " · Explanation covers only the first 128 tokens"
          : ""}
      </p>
    </section>
  );
}
