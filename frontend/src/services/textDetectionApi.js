
const API_BASE_URL = "http://127.0.0.1:8000";

export async function detectText(text) {
  const response = await fetch(
    `${API_BASE_URL}/api/text/detect`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ text }),
    }
  );

  if (!response.ok) {
    let message = `Request failed: ${response.status}`;

    try {
      const error = await response.json();
      if (typeof error.detail === "string") {
        message = error.detail;
      }
    } catch {
      // Use the default error message.
    }

    throw new Error(message);
  }

  return response.json();
}
