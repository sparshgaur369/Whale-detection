# MailWebAI — System Design Architecture

> Comprehensive system design covering infrastructure, data flow, caching, rate limiting, AI/ML pipeline, billing, and deployment.

---

## 1. High-Level System Architecture

```mermaid
graph TB
    subgraph Client["🖥️ Client Layer"]
        Browser["Browser (React 18)"]
        Landing["Landing Page"]
        MailUI["Mail Dashboard"]
        AI["AI Chatbot"]
    end

    subgraph Edge["🛡️ Edge / Middleware"]
        Clerk["Clerk Auth Middleware"]
    end

    subgraph API["⚡ API Layer (Next.js 14)"]
        TRPC["tRPC Router"]
        REST["REST API Routes"]
    end

    subgraph Services["🔧 Services Layer"]
        EmailSync["Email Sync Engine"]
        AIEngine["AI / ML Engine"]
        Billing["Billing Service"]
        RateLimiter["Rate Limiting"]
        Cache["Cache Layer"]
    end

    subgraph External["☁️ External Services"]
        Aurinko["Aurinko API"]
        OpenAI["OpenAI API"]
        Stripe["Stripe API"]
        ClerkSvc["Clerk Service"]
        S3["AWS S3"]
    end

    subgraph Data["💾 Data Layer"]
        Postgres["PostgreSQL (Neon)"]
        Redis["Redis (Upstash)"]
        Orama["Orama Vector Index"]
    end

    Browser --> Clerk --> API
    TRPC --> Services
    REST --> Services
    EmailSync --> Aurinko
    AIEngine --> OpenAI
    Billing --> Stripe
    Cache --> Redis
    Services --> Postgres
    EmailSync --> Orama
    Services --> S3
    Clerk --> ClerkSvc
```

---

## 2. Request Flow & Authentication

```mermaid
sequenceDiagram
    participant U as User Browser
    participant M as Clerk Middleware
    participant R as Next.js Route
    participant DB as PostgreSQL

    U->>M: HTTP Request
    
    alt Public Route (/, /sign-in, /api/webhooks, /api/stripe)
        M->>R: Pass through (no auth)
    else Protected Route
        M->>M: auth().protect()
        alt No valid session
            M-->>U: 401 Redirect to /sign-in
        else Valid session
            M->>R: Forward with userId
        end
    end

    R->>R: Per-User Rate Limit (30 RPM)
    
    alt Rate limit OK
        R->>DB: Process request
        DB-->>R: Response
        R-->>U: 200 OK
    else Over limit — queued
        R->>R: Poll every 500ms (max 15s)
        alt Slot opens
            R->>DB: Process request
            R-->>U: 200 OK
        else Timeout
            R-->>U: 429 Too Many Requests
        end
    end
```

---

## 3. Data Model (ER Diagram)

```mermaid
erDiagram
    User {
        string id PK
        string emailAddress UK
        string firstName
        string lastName
        string imageUrl
        string stripeSubscriptionId FK
        enum role
    }

    Account {
        string id PK
        string userId FK
        json binaryIndex
        string token UK
        string provider
        string emailAddress
        string name
        string nextDeltaToken
    }

    Thread {
        string id PK
        string subject
        datetime lastMessageDate
        string accountId FK
        boolean done
        boolean inboxStatus
        boolean draftStatus
        boolean sentStatus
    }

    Email {
        string id PK
        string threadId FK
        datetime sentAt
        string subject
        string body
        string bodySnippet
        boolean hasAttachments
        enum emailLabel
        string fromId FK
    }

    EmailAddress {
        string id PK
        string name
        string address
        string accountId FK
    }

    EmailAttachment {
        string id PK
        string name
        string mimeType
        int size
        string emailId FK
    }

    StripeSubscription {
        string id PK
        string userId FK
        string subscriptionId UK
        string customerId
        datetime currentPeriodEnd
    }

    ChatbotInteraction {
        string id PK
        string day
        int count
        string userId FK
    }

    User ||--o{ Account : owns
    User ||--o| StripeSubscription : subscribes
    User ||--o| ChatbotInteraction : tracks
    Account ||--o{ Thread : contains
    Account ||--o{ EmailAddress : has
    Thread ||--o{ Email : contains
    Email }o--|| EmailAddress : from
    Email }o--o{ EmailAddress : to
    Email }o--o{ EmailAddress : cc
    Email ||--o{ EmailAttachment : has
```

---

## 4. Email Sync Pipeline

```mermaid
flowchart TB
    subgraph Trigger["Trigger Events"]
        Init["Initial Sync<br/>(POST /api/initial-sync)"]
        Webhook["Aurinko Webhook<br/>(POST /api/aurinko/webhook)"]
    end

    subgraph Dedup["Deduplication"]
        SyncDedup["shouldProcessSync()<br/>Redis SET NX — TTL 5min"]
    end

    subgraph Sync["Sync Engine"]
        StartSync["startSync(daysWithin)"]
        CreateSub["createSubscription()"]
        PerfSync["performInitialSync()"]
        GetUpdated["getUpdatedEmails(deltaToken)"]
        SyncEmails["syncEmails() — paginated"]
    end

    subgraph Processing["Data Processing"]
        UpsertEmail["upsertEmail() — Prisma upsert"]
        UpsertAddr["upsertEmailAddress()"]
        UpsertAttach["upsertAttachment()"]
        PLimiter["p-limit concurrency"]
    end

    subgraph Indexing["Search Indexing"]
        OramaIdx["Orama Index Update"]
        Embeddings["OpenAI Embeddings<br/>(ada-002)"]
        Turndown["HTML to Markdown"]
    end

    subgraph CacheInv["Cache Invalidation"]
        InvThreads["invalidateThreadCaches()"]
    end

    Init --> PerfSync --> StartSync --> GetUpdated --> SyncEmails
    Init --> CreateSub
    Webhook --> SyncDedup
    SyncDedup -->|First caller| SyncEmails
    SyncDedup -->|Duplicate| Skip["Skip"]
    SyncEmails --> PLimiter
    PLimiter --> UpsertEmail --> UpsertAddr
    UpsertEmail --> UpsertAttach
    PLimiter --> OramaIdx --> Turndown
    OramaIdx --> Embeddings
    UpsertEmail --> InvThreads
```

---

## 5. AI / ML Architecture

```mermaid
flowchart TB
    subgraph Input["User Input"]
        Chat["Chat Message"]
        Complete["Text Completion"]
        Search["AI Search"]
    end

    subgraph PerUserRL["Per-User Rate Limit"]
        UPRL["acquireUserRateLimit()<br/>30 RPM / user — 15s queue"]
    end

    subgraph RAG["RAG Pipeline"]
        OramaSearch["Orama Vector Search"]
        EmbedQuery["Query to Embedding"]
        HybridSearch["Hybrid Search<br/>similarity >= 0.50"]
        Context["Context Assembly"]
    end

    subgraph OpenAIRL["OpenAI Rate Limiter"]
        Acquire["acquire(priority, tokens)"]
        SlidingWindow["Redis Sliding Window<br/>5000 RPM / 2M tokens"]
        Queue["Priority FIFO Queue"]
    end

    subgraph Models["OpenAI Models"]
        GPT4Nano["gpt-4.1-nano"]
        Ada002["text-embedding-ada-002"]
    end

    subgraph Tools["Tool Calling"]
        SearchEm["search_emails"]
        OpenEm["open_email"]
    end

    Chat --> UPRL --> RAG
    Complete --> UPRL --> OpenAIRL
    Search --> UPRL
    RAG --> EmbedQuery --> HybridSearch
    OramaSearch --> HybridSearch --> Context --> OpenAIRL
    OpenAIRL --> Acquire --> SlidingWindow
    Acquire --> Queue --> Models
    GPT4Nano --> Tools
    GPT4Nano --> Stream["Streaming Response"]
    Tools --> SearchEm
    Tools --> OpenEm
```

---

## 6. Caching Strategy

```mermaid
flowchart LR
    subgraph Req["Requests"]
        R1["getThreads()"]
        R2["getThread(id)"]
        R3["getThreadCount()"]
        R4["getAccounts()"]
        R5["getSubscriptionStatus()"]
    end

    subgraph Redis["Redis Cache"]
        C1["threads:{acct}:{tab}:{done}<br/>TTL 30s"]
        C2["thread:{acct}:{threadId}<br/>TTL 60s"]
        C3["threadcount:{acct}:{tab}<br/>TTL 30s"]
        C4["accounts:{userId}<br/>TTL 300s"]
        C5["sub:status:{userId}<br/>TTL 300s"]
    end

    subgraph DB["PostgreSQL"]
        PG["Prisma"]
    end

    R1 -->|HIT| C1
    R1 -->|MISS| PG -->|SET| C1
    R2 -->|HIT| C2
    R2 -->|MISS| PG -->|SET| C2
    R3 -->|HIT| C3
    R3 -->|MISS| PG -->|SET| C3
    R4 -->|HIT| C4
    R4 -->|MISS| PG -->|SET| C4
    R5 -->|HIT| C5
    R5 -->|MISS| PG -->|SET| C5
```

### Cache Invalidation Triggers

| Event | Keys Invalidated |
|---|---|
| Email sync | `threads:*`, `thread:*`, `threadcount:*` for account |
| Thread done/undone | `threads:*`, `thread:*`, `threadcount:*` for account |
| Mark as read | `thread:{id}` + `threads:*` lists |
| Stripe webhook | `sub:status:{userId}` |

---

## 7. Rate Limiting Architecture

```mermaid
flowchart TB
    subgraph L1["Layer 1: Per-User (30 RPM)"]
        UserReq["Request"] --> CheckUser["Redis ZCARD<br/>user:rpm:{userId}"]
        CheckUser --> Under30{"< 30?"}
        Under30 -->|Yes| RecordUser["ZADD + EXPIRE 120s"]
        Under30 -->|No| Wait["Poll 500ms<br/>max 15s"]
        Wait --> CheckUser
        Wait -->|Timeout| Reject["429"]
    end

    subgraph L2["Layer 2: OpenAI API (5000 RPM)"]
        RecordUser --> CheckAPI["Sliding Window Check"]
        CheckAPI --> CanProceed{"Under limits?"}
        CanProceed -->|Yes| Record["Record + Call OpenAI"]
        CanProceed -->|No| PQ["Priority Queue<br/>max 100 / 30s timeout"]
        PQ --> CheckAPI
    end

    subgraph Fallback["Redis Unavailable"]
        InMem["In-Memory Fallback<br/>(Map + Array)"]
    end

    CheckUser -.-> InMem
    CheckAPI -.-> InMem
```

### Priority Levels

| Priority | Used By |
|---|---|
| `high` | Reserved |
| `normal` | Chat responses |
| `low` | Completion, embeddings |

---

## 8. API Route Map

```mermaid
flowchart LR
    subgraph Public["Public (No Auth)"]
        WH["/api/aurinko/webhook — Sync trigger"]
        SW["/api/stripe — Stripe webhooks"]
        IS["/api/initial-sync — Bootstrap sync"]
    end

    subgraph Protected["Protected (Clerk)"]
        Chat["/api/chat — AI chatbot"]
        AIS["/api/ai-search — Email search"]
        Comp["/api/completion — Autocomplete"]
        Upload["/api/upload — S3 presigned URL"]
        AuCB["/api/aurinko/callback — OAuth"]
    end

    subgraph tRPC["tRPC (Protected)"]
        MR["mail.*<br/>getAccounts, getThreads,<br/>getThread, sendEmail,<br/>setDone, markAsRead,<br/>search, syncEmails"]
        WHR["webhooks.*<br/>get/create/deleteWebhook"]
    end
```

---

## 9. Frontend Component Architecture

```mermaid
graph TB
    subgraph Root["Root Layout"]
        ClerkP["ClerkProvider"]
        ThemeP["ThemeProvider"]
        TRPCP["tRPC + React Query"]
    end

    subgraph Landing["Landing Page"]
        Navbar --> ScrollSeq["ScrollSequence"]
        ScrollSeq --> HowItWorks
        HowItWorks --> Pricing
        Pricing --> Footer
    end

    subgraph Mail["Mail Dashboard"]
        MailLayout["Mail Layout"]
        Sidebar["Sidebar"]
        AccountSwitcher
        AskAI["AskAI Chat"]
        ComposeBtn["ComposeButton"]
        ThreadList
        ThreadDisplay
        EmailDisplay
        ReplyBox
        FilterBar
        SearchBar
    end

    subgraph State["State"]
        Jotai["Jotai Atoms"]
        RQ["React Query"]
        AISDK["AI SDK useChat"]
    end

    Root --> Landing
    Root --> Mail
    MailLayout --> Sidebar
    Sidebar --> AccountSwitcher
    Sidebar --> AskAI
    Sidebar --> ComposeBtn
    MailLayout --> ThreadList
    MailLayout --> ThreadDisplay --> EmailDisplay
    ThreadDisplay --> ReplyBox
    Mail --> State
```

---

## 10. Deployment & Infrastructure

```mermaid
flowchart LR
    subgraph Dev["Development"]
        Code["Source Code"] --> Git["Git Push"]
    end

    subgraph CI["CI/CD"]
        Git --> CB["Cloud Build"]
        CB --> Docker["Docker Build<br/>node:20"]
        Docker --> GCR["GCR<br/>gcr.io/mailwebai"]
    end

    subgraph Runtime["Production"]
        GCR --> CR["Cloud Run"]
        LB["Load Balancer"] --> CR
    end

    subgraph Services["Managed Services"]
        Neon["Neon PostgreSQL"]
        Upstash["Upstash Redis"]
        ClerkS["Clerk Auth"]
        StripeS["Stripe Billing"]
        OpenAIS["OpenAI"]
        AurinkoS["Aurinko Email"]
        S3S["AWS S3"]
    end

    CR --> Neon & Upstash & ClerkS & StripeS & OpenAIS & AurinkoS & S3S
```

---

## 11. Billing & Subscription Flow

```mermaid
sequenceDiagram
    participant U as User
    participant App as Next.js
    participant S as Stripe
    participant DB as PostgreSQL
    participant R as Redis

    U->>App: Click "Upgrade"
    App->>S: checkout.sessions.create($29/mo)
    S-->>U: Redirect to Checkout
    U->>S: Pay
    S-->>U: Redirect to /mail

    S->>App: POST /api/stripe (webhook)
    App->>DB: Upsert StripeSubscription
    App->>R: Invalidate sub:status:{userId}

    Note over U,R: Access Control (every AI request)
    U->>App: Chat message
    App->>R: Check sub:status:{userId}
    alt Free (15/day limit)
        App->>R: INCR chatbot:daily:{userId}:{date}
    else Premium (unlimited)
        App->>App: No limit
    end
    App-->>U: Stream response
```

---

## 12. Search Architecture

```mermaid
flowchart TB
    subgraph Input["Search Input"]
        NL["Natural Language<br/>(chat query)"]
        Structured["Structured Filters<br/>(from, date, keyword)"]
    end

    subgraph RAG["Vector Search Path"]
        Embed["Embed query → 1536-dim"]
        Hybrid["Orama Hybrid Search<br/>text + vector, sim >= 0.50"]
        Index["Account.binaryIndex<br/>(JSON-serialized Orama)"]
        Results["Top-10 contexts"]
    end

    subgraph Prisma["Structured Search Path"]
        Parse["Parse filters"]
        Where["Build ThreadWhereInput"]
        Query["Prisma findMany<br/>(max 20 threads)"]
    end

    NL --> Embed --> Hybrid
    Index --> Hybrid --> Results
    Structured --> Parse --> Where --> Query
```

---

## 13. Redis Key Space

| Key Pattern | Purpose | TTL |
|---|---|---|
| `threads:{acct}:{tab}:{done}` | Thread list cache | 30s |
| `thread:{acct}:{threadId}` | Single thread | 60s |
| `threadcount:{acct}:{tab}` | Thread count | 30s |
| `accounts:{userId}` | User accounts | 300s |
| `sub:status:{userId}` | Subscription status | 300s |
| `chatbot:daily:{userId}:{date}` | Daily AI usage | Until midnight |
| `ratelimit:openai:requests` | Global RPM tracker | 60s |
| `ratelimit:openai:tokens` | Global token tracker | 60s |
| `user:rpm:{userId}` | Per-user RPM | 120s |
| `sync:dedup:{acct}:{eventId}` | Webhook dedup | 300s |

---

## 14. Error Handling & Resilience

```mermaid
flowchart TB
    subgraph Patterns["Resilience Patterns"]
        GF["🔄 Graceful Degradation<br/>Redis down → in-memory"]
        RLQ["⏳ Queue-Based Rate Limiting<br/>Hold, don't reject"]
        DD["🔁 Webhook Dedup<br/>SET NX idempotency"]
        CA["📦 Cache-Aside<br/>Read → miss → DB → populate"]
        IU["🆔 Idempotent Upserts<br/>Prisma upsert for sync"]
    end

    subgraph Errors["HTTP Errors"]
        E401["401 Unauthorized"]
        E404["404 Not Found"]
        E429["429 Rate Limited<br/>(Retry-After header)"]
        E500["500 Server Error"]
    end
```

---

## 15. Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Framework** | Next.js 14 (App Router) | SSR, API routes |
| **Language** | TypeScript | Type safety |
| **Styling** | Tailwind CSS | Utility CSS |
| **UI** | Radix UI + shadcn/ui | Component primitives |
| **Animation** | Framer Motion | Landing animations |
| **State** | Jotai + React Query | Client + server state |
| **API** | tRPC v11 + REST | Type-safe RPC + streaming |
| **Auth** | Clerk | OAuth, sessions |
| **Database** | PostgreSQL (Neon) | Primary store |
| **ORM** | Prisma v5 | DB access |
| **Cache** | Upstash Redis | Caching + rate limiting |
| **Email** | Aurinko API | Gmail / Office365 |
| **AI** | OpenAI GPT-4.1-nano / ada-002 | Chat, completions, embeddings |
| **AI SDK** | Vercel AI SDK v3 | Streaming, tools |
| **Search** | Orama | Hybrid vector search |
| **Billing** | Stripe | Subscriptions |
| **Storage** | AWS S3 | File uploads |
| **Deploy** | Cloud Run + Cloud Build | Container hosting, CI/CD |
| **Container** | Docker (node:20) | Containerization |
| **Editor** | Novel (Tiptap) | Rich text composition |
