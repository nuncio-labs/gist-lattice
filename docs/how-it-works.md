# How GistLattice Works: A Human Example

To truly understand how GistLattice's architecture functions, it helps to look at a plain-English example of a user interacting with an AI backed by our library.

Let's follow a user named **Sarah** chatting with her AI assistant, **Nexus**.

---

### Day 1: The Major Life Event

**Sarah:** "I just got hired at Acme Corp and I'm moving to New York next month! I'm honestly incredibly stressed about finding an apartment."  
**Nexus:** "Congratulations on Acme Corp! Finding an apartment in New York is tough, but we can break it down step-by-step. Don't stress, I'll help you."

**Behind the scenes:**  
Sarah gets that response instantly because Nexus used its *Short-Term Buffer* (a standard sliding window of the last few chat messages). But the moment the message sends, GistLattice quietly goes to work in the background. It reflects on the chat and extracts:
- **Gist:** "Sarah was hired at Acme Corp, is moving to New York, and needs an apartment."
- **Importance:** `0.9` *(This is a major life event, do not forget this).*
- **Valence:** `-0.7` *(She is highly stressed/anxious).*
- **Semantic Graph Updates:** It updates her internal knowledge graph setting her `LOCATED_AT` to "New York" and her `CURRENT_STATE` to "Stressed".

---

### Day 2: The Short-Term Follow-up

**Sarah:** "Are there any updates on listings?"  
**Nexus:** "I found three apartments in Brooklyn that fit your budget."

**Behind the scenes:**  
GistLattice wasn't really needed here! Because this conversation happened yesterday, the exact chat logs are still sitting in Nexus's *Short-Term Buffer*. Nexus remembers what "listings" means because it literally just talked about it.

---

### Six Months Later: The Magic of GistLattice

Sarah hasn't used Nexus for half a year. The Short-Term Buffer was wiped out months ago. As far as standard AI is concerned, Sarah is a blank slate.

**Sarah:** "I'm exhausted from work today. Any ideas for what I should do for dinner?"

**Behind the scenes:**  
Before sending Sarah's message to the LLM, the system calls `memory.hydrate_context()`. GistLattice reaches deep into the Episodic vector database and the Semantic Neo4j graph, realizes who Sarah is, and secretly injects this block at the top of the LLM's prompt:

> *[SYSTEM CONTEXT: The user is Sarah. Her current location is New York. She is employed at Acme Corp. Past highly important memory: She was very stressed about moving here 6 months ago.]*

The LLM reads that hidden context and generates its response.

**Nexus:** "Rough day at Acme Corp? Since you're in New York, you should treat yourself. There's a fantastic, highly-rated Italian place just two blocks from your office that is perfect for unwinding. You've survived the stressful move, you deserve a break!"

**Sarah is blown away.** She hasn't mentioned Acme Corp or New York in *six months*, but her AI remembered her exact life situation and even referenced how stressed she used to be about the move.

That is the power of GistLattice!
