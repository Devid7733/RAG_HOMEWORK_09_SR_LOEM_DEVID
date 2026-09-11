# Reflection

The strongest part of this project was seeing the complete RAG pipeline answer
practical IT-support questions from three different guides. The
`nomic-embed-text` model kept the most relevant document among the leading
results for every on-topic query. Using the retrieved context, `llama3.2`
correctly recommended WPA2-PSK or WPA3-PSK instead of WEP or WPA, identified
the 12-character minimum wireless password, explained that a jammed printer
should first be turned off, and advised using manual or advanced setup when
automatic Android email configuration fails. For the vacation-policy question,
the model correctly stated that the documents did not contain enough
information rather than answering from its general knowledge.

The harder part was keeping the persisted vector index synchronized with the
source folder. After replacing the original documents, an existing ChromaDB
collection could still contain vectors from older files unless the index was
explicitly rebuilt. Adding a reindex option to the test workflow made the
relationship between ingestion and retrieval clearer and ensured that the
recorded results matched the current documents.

A useful future improvement would be reranking. Although the printer query
retrieved only printer-related chunks in its top three results, the first chunk
was an additional-tips passage while the chunk containing Step 1 ranked second.
A reranker could reorder those candidates before generation, placing the most
direct answer first and reducing distracting context as the collection grows.
