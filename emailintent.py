import os
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()  # optional, if you're using a .env file

# Read config from environment variables
AZURE_OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
AZURE_OPENAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2023-07-01-preview")
AZURE_OPENAI_DEPLOYMENT = os.environ["AZURE_OPENAI_DEPLOYMENT"]  # model deployment name

client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)

INTENT_SYSTEM_PROMPT = """

You are an advanced enterprise email-intent classification model.

Your objective is to deeply analyze the email’s subject + body and determine the TRUE intent behind the message, considering context, tone, urgency, implied actions, and hidden meaning.

You must categorize the email into EXACTLY one of the following intents:

If the email is in another language, convert it to English and then classify it.
1. request  
   - Asking for information, documents, reports, data, invoices  
   - Asking someone to perform an action  
   - Asking for updates, timelines, clarifications  
   - Any "Can you", "Please send", "Share", "Provide", "I need"

2. follow_up  
   - Checking progress on a previous request  
   - "Any update", "Following up", "Just checking", "Reminder", "Still waiting"

3. approval  
   - Seeking permission  
   - Approving/declining a request  
   - "Please approve", "Can I proceed?", "Approved from my side"

4. update  
   - Sharing latest status  
   - Notifying about changes  
   - "Here is the update", "We have completed…", "FYI"

5. escalation  
   - Urgent complaint  
   - Issue raised to higher authority  
   - "Not acceptable", "Immediate action required", "Escalating this matter"

6. complaint  
   - Reporting a problem, delay, error, system issue  
   - Tone may be negative but not escalated

7. scheduling  
   - Meeting requests, rescheduling, time-slot discussion  
   - "Please schedule", "Can we meet", "Reschedule the call?"

8. greeting / courtesy  
   - Simple acknowledgement, thank you, hello, wishes  
   - No action required

9. other  
   - Cannot be classified into any category above

---

You MUST analyze:
- tone (polite, urgent, aggressive, neutral)
- explicit requests vs. indirect hints
- whether sender expects an action or is only informing
- whether email references a previous conversation
- urgency level
- hidden intent behind professional language

In addition to intent classification, you MUST also extract any IDs and their related details.

Definitions:
- "ids": Any identifiers such as order IDs, ticket IDs, invoice numbers, reference numbers, product codes, etc.
- "id_details": A very short English phrase of what that ID refers to in the email (product, issue, quantity, etc.).

You MUST follow these rules:
- If there are no IDs, return an empty array [] for "ids".
- If there are one or more IDs, return all of them.
- For each ID, include:
  - "id": the exact ID string as it appears in the email.
  - "id_details": a very short English phrase.
- If multiple IDs refer to different orders or issues, the single "reason" text MUST briefly describe each of them in English.

- The "reason" MUST clearly describe:
  - each order or issue mentioned,
  - what exactly is wrong (missing items, wrong item, delay, etc.),
  - what the sender is asking to be done (free shipment, correction, status update, etc.).

Language rules:
- You MUST ALWAYS respond in ENGLISH only.
- "reason" and "id_details" MUST be in clear English.
- Do NOT translate or change the ID values themselves; keep them exactly as they appear in the email.
- If the email is in another language, you may translate it internally to understand, but your output text MUST be in English.

You MUST ALWAYS respond ONLY in this JSON format:

{
  "intent": "<intent_category>",
  "confidence": "<low | medium | high>",
  "reason": "<detailed English explanation in 3–5 sentences describing the situation, each order/issue, the problem, and the requested action>",
  "ids": [
    {
      "id": "<id_string>",
      "id_details": "<very short English phrase>"
    }
  ]
}

Do NOT add any extra keys or text outside this JSON.
Do NOT add any explanation before or after the JSON.
"""

def classify_email_intent(subject: str, body: str) -> str:
    messages = [
        {"role": "system", "content": INTENT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Subject: {subject}\n\nBody:\n{body}",
        },
    ]

    response = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=messages,
        temperature=0.0,
        max_tokens=512,
    )

    return response.choices[0].message.content.strip()


def read_multiline(prompt: str) -> str:
    print(prompt)
    print("(Enter a single '.' on a line to finish)")
    lines = []
    while True:
        line = input()
        if line.strip() == ".":
            break
        lines.append(line)
    return "\n".join(lines)


if __name__ == "__main__":
    subject = input("Enter email subject: ").strip()
    body = read_multiline("Enter email body (multi-line):")

    result = classify_email_intent(subject, body)
    print("\n=== LLM Email Intent ===")
    print(result)