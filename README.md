# LangGraph-Reflection

This prebuilt graph is an agent that uses a reflection-style architecture to check and improve an initial agent's output.

This reflection agent uses two subagents:
- A "main" agent, which is the agent attempting to solve the users task
- A "critique" agent, which checks the main agents work and offers any critiques

The reflection agent has the following architecture:

1. First, the main agent is called
2. Once the main agent is finished, the critique agent is called
3. Based on the result of the critique agent:
   - If the critique agent finds something to critique, then the main agent is called again
   - If there is nothing to critique, then the overall reflection agent finishes
4. Repeat until the overall reflection agent finishes


We make some assumptions about the graphs:
- The main agent should take as input a list of messages
- The reflection agent should return a **user** message if there is any critiques, otherwise it should return **no** messages.

## Examples

Below are a few examples of how to use this reflection agent.

### LLM-as-a-Judge

In this example, the reflection agent uses another LLM to judge it's output



### Coding

Code is easy to "check" in deterministic ways. In this example, we use a linter to check Python code.

