# 🕵️‍♀️ The "Compose Email" Trace: Line-by-Line & Byte-by-Byte

You asked for the exact path, line numbers, and data formats. Here is the lifecycle of the command: **"Send an email to sparsh@gmail.com"**.

---

## 🟢 PHASE 1: The Trigger (Client -> Server)

**📍 Location:** `src/app/mail/components/ask-ai.tsx`
**Lines:** 32-34

When you press "Enter", the `useChat` hook (from Vercel AI SDK) takes your input and fires a `POST` request.

**The Code (Lines 32-34):**
```typescript
const { input, handleInputChange, handleSubmit } = useChat({
    api: "/api/chat", // <--- The destination
    body: { accountId }, // <--- Extra data we send
    // ...
});
```

**📨 The JSON Payload (What actually travels over the internet):**
```json
{
  "messages": [
    {
      "role": "user",
      "content": "Send an email to sparsh@gmail.com"
    }
  ],
  "accountId": "acc_12345",
  "threadId": "thread_67890"  // (If you are viewing an email)
}
```

---

## 🔵 PHASE 2: The Brain (Server Logic)

**📍 Location:** `src/app/api/chat/route.ts`
**Lines:** 23, 66, 146

The Server receives that JSON.

1.  **Line 23:** `export async function POST(req: Request)` — The entry point.
2.  **Line 66:** `const { messages, accountId } = await req.json();` — It unpacks the JSON above.
3.  **Line 146:** It calls `rateLimitedStreamText()` to talk to OpenAI.

**🧠 The OpenAI Request (What the Server sends to GPT):**
The server wraps your simple message in a huge "System Prompt" (Line 116).
```json
{
  "model": "gpt-4o-mini",
  "messages": [
    {
      "role": "system",
      "content": "You are an AI assistant... THE TIME NOW IS..."
    },
    {
      "role": "user",
      "content": "Send an email to sparsh@gmail.com"
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "compose_email",
        "description": "Open the compose email view...",
        "parameters": {
          "type": "object",
          "properties": {
            "to": { "type": "array", "items": { "type": "string" } },
            "subject": { "type": "string" },
            "body": { "type": "string" }
          },
          "required": ["to", "subject", "body"]
        }
      }
    }
  ]
}
```

---

## 🟣 PHASE 3: The Decision (OpenAI -> Server)

OpenAI's math models run. They decide: *"Ah, the user wants to email Sparsh. I have a tool for that called `compose_email`."*

**The Response (JSON from OpenAI):**
```json
{
  "id": "call_abc123",
  "choices": [
    {
      "message": {
        "role": "assistant",
        "tool_calls": [
          {
            "id": "call_abc123",
            "type": "function",
            "function": {
              "name": "compose_email",
              "arguments": "{\"to\":[\"sparsh@gmail.com\"], \"subject\": \"(No Subject)\", \"body\": \"Hi Sparsh,\"}"
            }
          }
        ]
      }
    }
  ]
}
```

---

## 🟠 PHASE 4: The Stream (Server -> Client)

**📍 Location:** Network Stream
The server doesn't wait for the whole thing. It streams this back to your browser using the **Vercel AI SDK Protocol**.

**The Stream Data:**
It sends chunks that look like this (simplified):
1.  `9:{"toolCallId":"call_abc123","toolName":"compose_email","args":{"to":["sparsh@gmail.com"],"subject":"","body":"Hi Sparsh,"}}`

*(The prefix `9:` tells the SDK "This is a tool call, not just text".)*

---

## 🔴 PHASE 5: The Execution (Client UI)

**📍 Location:** `src/app/mail/components/ask-ai.tsx`
**Lines:** 45-60

Back in the browser, the `useChat` hook sees that `9:` prefix and triggers the `onToolCall` function.

**The Code (Lines 45-60):**
```typescript
onToolCall: async ({ toolCall }) => {
    // 1. Check the name
    if (toolCall.toolName === 'compose_email') {
        
        // 2. Create the ACTION object
        const action = {
            type: 'compose_email',
            to: ["sparsh@gmail.com"],
            subject: "",
            body: "Hi Sparsh,"
        };

        // 3. Dispatch to Jotai (Global State)
        setAIAction(action); 
        
        // 4. Show feedback
        toast.info('Opening compose view...');
    }
}
```

---

## ⚫ PHASE 6: The Reaction (Compose Button)

**📍 Location:** `src/app/mail/components/compose-button.tsx`
**Lines:** 97-102

The `ComposeButton` is listening.

**The Code (Lines 97-102):**
```typescript
React.useEffect(() => {
    // 1. Is it a compose action?
    if (aiAction?.type !== 'compose_email') return;

    // 2. Open the Drawer
    setOpen(true); 

    // 3. Fill the fields (from the JSON args we got in Phase 4)
    setToValues(aiAction.to...);
    setSubject(aiAction.subject);
    setAiBody(aiAction.body);

}, [aiAction]); // <--- Runs whenever 'aiAction' changes
```

**✅ Result:** The drawer slides open, "sparsh@gmail.com" appears in the To field, and the AI starts ghost-typing the body.
