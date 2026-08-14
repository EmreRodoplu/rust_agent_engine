use uuid::Uuid;
use super::{Chunk, Chunker, Document};

pub struct RecursiveChunker {
    pub chunk_size: usize,   
    pub chunk_overlap: usize, 
}

impl RecursiveChunker {
    pub fn new(chunk_size: usize, chunk_overlap: usize) -> Self {
        Self {
            chunk_size,
            chunk_overlap,
        }
    }

    
    fn split_text(&self, text: &str) -> Vec<String> {
        let mut chunks = Vec::new();
        let separators = ["\n\n", "\n", ". ", " ", ""];

        self.split_recursive(text, &separators, &mut chunks);
        
        chunks
    }

    fn split_recursive(&self, text: &str, separators: &[&str], chunks: &mut Vec<String>) {
        if text.len() <= self.chunk_size {
            chunks.push(text.to_string());
            return;
        }

        
        let separator = separators.iter()
            .find(|&&sep| sep.is_empty() || text.contains(sep))
            .unwrap_or(&"");

        let mut splits = if separator.is_empty() {
            text.chars().map(|c| c.to_string()).collect::<Vec<String>>()
        } else {
            text.split(separator).map(|s| s.to_string()).collect::<Vec<String>>()
        };

        let mut current_chunk = String::new();
        let mut i = 0;

        while i < splits.len() {
            let split = &splits[i];
            
            let potential_len = if current_chunk.is_empty() {
                split.len()
            } else {
                current_chunk.len() + separator.len() + split.len()
            };

            if potential_len <= self.chunk_size {
                if !current_chunk.is_empty() {
                    current_chunk.push_str(separator);
                }
                current_chunk.push_str(split);
                i += 1;
            } else {
                if !current_chunk.is_empty() {
                    chunks.push(current_chunk.clone());
                    if self.chunk_overlap > 0 {
                        let overlap_start = current_chunk.len().saturating_sub(self.chunk_overlap);
                        current_chunk = current_chunk[overlap_start..].to_string();
                    } else {
                        current_chunk.clear();
                    }
                } else {
                    if separators.len() > 1 {
                        let next_separators = &separators[1..];
                        self.split_recursive(split, next_separators, chunks);
                    } else {
                        let (head, tail) = split.split_at(self.chunk_size);
                        chunks.push(head.to_string());
                        splits.insert(i + 1, tail.to_string());
                    }
                    i += 1;
                    current_chunk.clear();
                }
            }
        }

        if !current_chunk.is_empty() {
            chunks.push(current_chunk);
        }
    }
}

impl Chunker for RecursiveChunker {
    fn chunk(&self, document: &Document) -> anyhow::Result<Vec<Chunk>> {
        let texts = self.split_text(&document.content);
        
        let chunks: Vec<Chunk> = texts.into_iter().map(|text| {
            Chunk {
                id: Uuid::new_v4().to_string(), 
                document_id: document.id.clone(),
                content: text,
                embedding: None, 
            }
        }).collect();

        Ok(chunks)
    }
}