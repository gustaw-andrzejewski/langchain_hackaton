# Langchain Hackathon Project: GitHub Repo Query Tool

During an AI lang chain hackathon, I crafted this modest terminal app to delve into GitHub repositories and interact with OpenAI language models. Supply it with a GitHub repo URL, and it'll fetch the repo, transform it into a vector store, and enable you to pose questions about the code using OpenAI. The core action happens in `main.py`.

## Core Utilization

* Learning new libraries
* Navigating a large project

The tool is a nifty way to explore the content of a GitHub repository through natural language queries, bridging the gap between codebases and human language. It's not just about fetching a repo; it's about diving into the code, understanding its structure, and fetching insights straight from a natural language interface.

## Future Insights

* User Interface (managing query history, costs, etc.)
* Fetching repositories from other platforms (GitLab, etc.)
* Returning the source of answers
* Utilizing open-source models (from huggingface)
* Fetching repositories in multiple programming languages (+ maybe filtering search based on language?)
* Posing questions in multiple languages
* Possibility of retraining the chosen model on the code from the repo or through RLHF

The project opens doors to multiple enhancements that can significantly augment the interaction between users and code repositories. Through natural language processing, it lays the foundation for a more intuitive way to navigate and understand large codebases. Additionally, it hints at a future where not only can we ask questions to our code but also get insightful answers that drive our development forward.
