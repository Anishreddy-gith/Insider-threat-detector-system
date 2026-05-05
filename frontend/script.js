const API_BASE_URL = window.ITDS_API_BASE_URL || "https://insider-threat-api.onrender.com";
const TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJkZW1vLXByb2Zlc3NvciIsInVzZXJfaWQiOiJkZW1vLXByb2Zlc3NvciIsInJvbGUiOiJkZW1vIn0.NQjfB12I6MeibU0PF3OrNB_r10FroUEVVyhw5g2rzv4";

const userIdInput = document.getElementById("userId");
const output = document.getElementById("output");

function show(value) {
  output.textContent = typeof value === "string" ? value : JSON.stringify(value, null, 2);
}

async function parseResponse(response) {
  const text = await response.text();
  let payload;
  try {
    payload = JSON.parse(text);
  } catch {
    payload = text;
  }
  if (!response.ok) {
    throw payload;
  }
  return payload;
}

function assertConfigured() {
  if (API_BASE_URL.includes("YOUR_REAL_IP") || API_BASE_URL.includes("<EC2-IP>")) {
    throw "Replace API_BASE_URL in frontend/script.js with your Render backend URL.";
  }
}

document.getElementById("sendActivity").addEventListener("click", async () => {
  const userId = userIdInput.value.trim();
  if (!userId) {
    show("Enter a user_id.");
    return;
  }

  show("Sending activity...");
  try {
    assertConfigured();
    const response = await fetch(`${API_BASE_URL}/ingest`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${TOKEN}`,
      },
      body: JSON.stringify({
        user_id: userId,
        timestamp: new Date().toISOString(),
        activity_type: "logon",
        metadata: {},
      }),
    });
    show(await parseResponse(response));
  } catch (error) {
    show(error);
  }
});

document.getElementById("checkRisk").addEventListener("click", async () => {
  const userId = userIdInput.value.trim();
  if (!userId) {
    show("Enter a user_id.");
    return;
  }

  show("Checking risk...");
  try {
    assertConfigured();
    const response = await fetch(`${API_BASE_URL}/risk/${encodeURIComponent(userId)}`, {
      headers: {
        Authorization: `Bearer ${TOKEN}`,
      },
    });
    show(await parseResponse(response));
  } catch (error) {
    show(error);
  }
});
