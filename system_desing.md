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




# System Architecture

The following diagram illustrates the high-level architecture of the MailWebAI project, detailing the interactions between the client, external services, application layer, and data layer.

```mermaid
graph TD
    subgraph "Client Layer"
        User[User Browser]
    end

    subgraph "External Services"
        Clerk[Clerk Auth]
        Aurinko[Aurinko Email Engine]
        OpenAI[OpenAI LLM & Embeddings]
        Stripe[Stripe Payments]
        S3[AWS S3 Attachments]
    end

    subgraph "Application Layer - Next.js"
        AuthMiddleware["Middleware - Clerk"]
        TRPC[tRPC API Routes]
        Webhooks[Webhook Handlers]
        AICore[AI Assistant / RAG]
        SyncEngine[Sync Logic]
    end

    subgraph "Data Layer"
        Postgres[(PostgreSQL - Prisma)]
        Redis[(Redis - Upstash)]
        Orama[(Orama - Vector Search)]
    end

    %% Auth Flows
    User -->|Auth Redirect| Clerk
    Clerk -->|Session Token| AuthMiddleware
    AuthMiddleware -->|Protected Access| TRPC

    %% User Interactions
    User -->|View Threads/Mails| TRPC
    TRPC -->|Read/Write| Postgres
    TRPC -->|Cache Hit/Miss| Redis

    %% Sync Flows
    Aurinko -->|New Email Webhook| Webhooks
    Webhooks -->|Trigger Sync| SyncEngine
    SyncEngine -->|Store Emails| Postgres
    SyncEngine -->|Update Index| Orama
    SyncEngine -->|Invalidate Cache| Redis
    SyncEngine -->|Upload Attachments| S3

    %% AI Flows
    User -->|Ask Question| TRPC
    TRPC -->|Process Request| AICore
    AICore -->|Generate Embedding| OpenAI
    AICore -->|Vector Search| Orama
    AICore -->|Generate Response| OpenAI
    AICore -->|Context Retrieval| Postgres

    %% Payment Flows
    User -->|Subscribe| Stripe
    Stripe -->|Webhook| Webhooks
    Webhooks -->|Update Subscription| Postgres
```


## Key Components

1.  **Frontend (Client Layer)**:
    *   Built with **Next.js 14 App Router** using React Server Components.
    *   Interacts with the backend primarily via **tRPC**.

2.  **Authentication**:
    *   **Clerk**: Handles user authentication, session management, and secure access control.
    *   **Middleware**: Intercepts requests to ensure valid sessions before access to protected routes.

3.  **Email Engine**:
    *   **Aurinko**: Connects to email providers (Google, Office365) and synchronizes data.
    *   **Webhooks**: Received from Aurinko trigger real-time sync processes in the application.

4.  **Backend Services (Application Layer)**:
    *   **tRPC API Routes**: Type-safe API endpoints for client-server communication.
    *   **Sync Engine**: Handles the logic for processing incoming emails, storing them, and updating search indexes.
    *   **AI Core (RAG)**: Retrieves relevant context using vector search and generates responses via OpenAI.

5.  **Data & Storage**:
    *   **PostgreSQL (Prisma)**: Primary database for persistent storage (Users, Threads, Emails).
    *   **Redis (Upstash)**: Provides caching for performance (e.g., thread lists) and manages API rate limits.
    *   **Orama**: specialized vector database for semantic search across emails.
    *   **AWS S3**: Secure object storage for email attachments.

6.  **Payments**:
    *   **Stripe**: Manages subscriptions and payment processing. Webhooks update user subscription status in the database.
  
# Email Sync System Architecture

## 1. Overview
The Email Sync system in MailWebAI is designed to provide a real-time, bidirectional mirror of the user's email account. It leverages **Aurinko** as the unified email API provider (supporting Google, Office365, etc.) and employs a multi-stage synchronization pipeline to ensure data consistency, performance, and AI-readiness.

## 2. High-Level Architecture

The system operates in two distinct modes: **Initial Sync** (bootstrapping) and **Delta Sync** (real-time updates via webhooks).

```mermaid
graph TD
    subgraph "External Providers"
        Aurinko[Aurinko API]
        OpenAI[OpenAI API]
    end

    subgraph "Ingestion Layer"
        Webhook["Webhook Handler (/api/aurinko/webhook)"]
        InitialSync["Initial Sync Endpoint (/api/initial-sync)"]
        Dedup[Redis Deduplication]
    end

    subgraph "Processing Layer (Sync Engine)"
        Account[Account Class]
        SyncLoop[Pagination & Delta Loop]
        Turndown[HTML to Markdown]
        Embedding[Embedding Generator]
    end

    subgraph "Storage Layer"
        Postgres[(PostgreSQL - Prisma)]
        Orama[(Orama - Vector DB)]
        Redis[(Redis - Cache & Rate Limits)]
    end

    %% Flows
    Aurinko -->|Webhook Notification| Webhook
    Webhook -->|Check Event ID| Dedup
    Dedup -->|If New| Account

    InitialSync -->|Start Job| Account
    Account -->|Fetch Emails| Aurinko
    Account -->|Parse & Normalize| SyncLoop

    SyncLoop -->|Upsert Data| Postgres
    SyncLoop -->|Generate Vector| Turndown
    Turndown -->|Get Embedding| Embedding
    Embedding -->|Request| OpenAI
    Embedding -->|Store Vector| Orama

    SyncLoop -->|Invalidate Views| Redis
```

## 3. Core Components

### 3.1. Aurinko Integration (`src/lib/account.ts`)
*   **Provider**: Aurinko is used as the middleware to abstract IMAP/Graph API complexities.
*   **Authentication**: Uses Bearer tokens associated with the `Account` model in the database.
*   **Sync Strategy**:
    *   **Initial Sync**: Fetches emails from the last **3 days** (configurable). Polling mechanism checks for job completion (`ready` status).
    *   **Delta Sync**: Uses `deltaToken` to fetch only changes since the last sync.

### 3.2. Webhook Handling (`src/app/api/aurinko/webhook/route.ts`)
*   **Security**: Validates `X-Aurinko-Signature` using HMAC-SHA256 and the `AURINKO_SIGNING_SECRET`.
*   **Deduplication** (`src/lib/sync-dedup.ts`):
    *   Uses **Redis** (`setNX`) to prevent processing the same webhook event multiple times.
    *   Key format: `sync:dedup:{accountId}:{eventId}`.
    *   TTL: 5 minutes.
*   **Trigger**: Instantiates the `Account` class and triggers `syncEmails()`.

### 3.3. Sync Engine & Data Processing (`src/lib/sync-to-db.ts`)
The core logic resides in `syncEmailsToDatabase`. It processes batches of emails with controlled concurrency.

*   **Concurrency**: Uses `p-limit` to process up to **10 emails concurrently** during DB upserts.
*   **Address Normalization**:
    *   Extracts all unique email addresses (From, To, Cc, Bcc, ReplyTo) from a batch.
    *   Upserts them to the `EmailAddress` table first to ensure foreign key integrity.
*   **Thread Management**:
    *   Upserts `Thread` records given the `threadId` from Aurinko.
    *   Updates `lastMessageDate` and participant lists.
    *   **Folder Inference**: Determines if a thread is `inbox`, `sent`, or `draft` based on the labels of the emails contained within it.
*   **Email Storage**:
    *   Upserts `Email` records with full metadata (Subject, Body, snippet, etc.).
    *   Maps Aura-specific labels (e.g., `sysLabels`) to internal enums/booleans.

### 3.4. AI & Search Pipeline (`src/lib/sync-to-db.ts`, `src/lib/embeddings.ts`)
Every synced email is immediately prepared for RAG (Retrieval-Augmented Generation) and Semantic Search.

1.  **Text Extraction**: `turndown` library converts HTML email bodies to Markdown for cleaner LLM context.
2.  **Embedding Generation**:
    *   Uses `text-embedding-ada-002` via OpenAI.
    *   Request payload construction: `From: ... \n To: ... \n Subject: ... \n Body: ...`
    *   **Rate Limiting**: Embeddings are requested with `'low'` priority using the custom `OpenAIRateLimiter` to prevent saturating API quotas during bulk syncs.
3.  **Vector Storage**:
    *   **Orama**: A specialized vector database instance is initialized per user account (`OramaManager`).
    *   Inserts vectors alongside metadata (Thread ID, Timestamp) for hybrid search.

### 3.5. Caching & Invalidation (`src/lib/email-cache.ts`)
*   **Post-Sync**: Once a sync batch is complete, `invalidateThreadCaches(accountId)` is called.
*   **Mechanism**: Deletes Redis keys matching patterns for thread lists to ensure the frontend displays the latest data immediately.

## 4. Data Model (Prisma)

Key relationships enabling the sync architecture:

*   **Account**: Stores the `token` and `nextDeltaToken` (cursor).
*   **Thread**: Aggregates emails. Has computed flags (`inboxStatus`, `draftStatus`, `sentStatus`) for efficient querying.
*   **EmailAddress**: Normalized entity to allow graph-like queries (e.g., "all emails from X").
*   **Email**: The raw message unit. Stores `internetMessageId` for deduplication and `sysLabels` for folder categorization.

## 5. Error Handling & Resilience

*   **Webhook Retries**: Handled by Aurinko. Our deduplication logic ensures idempotency.
*   **Sync Failures**: 
    *   If `performInitialSync` fails, it logs the error and returns.
    *   Individual email processing errors (e.g., Prisma unique constraint race conditions) are caught and logged, preventing the entire batch from failing.
*   **Token Expiry**: Handled by the upstream `Account` class (not fully visible in sync logic but assumed handled during Aurinko API calls). 

## 6. Future Considerations / Scalability Limits
*   **Orama Persistence**: Currently, Orama indices might need to be persisted to disk or S3 to survive cold boots effectively if not using a cloud version.
*   **Queueing**: For extremely high usage, a dedicated message queue (e.g., BullMQ) between Webhook and Sync Engine would be better than `waitUntil`.


