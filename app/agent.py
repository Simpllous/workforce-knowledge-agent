# ruff: noqa
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import google.auth
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.models import Gemini
from google.genai import types

from .tools import store_knowledge, search_knowledge, list_sources

load_dotenv()

if not os.getenv("GOOGLE_CLOUD_PROJECT"):
    _, auth_project = google.auth.default()
    if auth_project:
        os.environ["GOOGLE_CLOUD_PROJECT"] = auth_project

os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")

_model = Gemini(
    model="gemini-flash-latest",
    retry_options=types.HttpRetryOptions(attempts=3),
)

capture_agent = Agent(
    name="capture_agent",
    model=_model,
    description="Conducts a structured knowledge-capture interview with an expert. Use when the user wants to record what someone knows before they leave.",
    instruction="""You are a knowledge capture specialist. Your job is to interview an experienced engineer or technician and extract their tacit knowledge before they leave.

Ask one clear question at a time. Cover these areas in order:
1. The core processes or procedures they own or know deeply
2. Common failures or problems they have solved, and how
3. Specifications, tolerances, or settings they keep in their head
4. Safety considerations or rules they follow that are not written down
5. Tips and shortcuts that only come from experience

After each answer, call store_knowledge with:
- content: the answer, self-contained and clearly worded
- source: the expert's name and role, e.g. "John Smith (Test Engineer)"
- category: one of process / troubleshooting / specification / safety / general

Confirm to the user that the answer was saved, then ask the next question.
Keep going until the user says they are done or there is nothing more to cover.
""",
    tools=[store_knowledge],
)

ingest_agent = Agent(
    name="ingest_agent",
    model=_model,
    description="Ingests a document (SOP, manual, NCR, email) into the knowledge base. Use when the user pastes or describes an existing document.",
    instruction="""You are a document ingestion specialist. Your job is to read documents and store their knowledge in a structured way.

When the user provides document text:
1. Read it carefully and identify distinct, self-contained pieces of knowledge
2. Store each piece separately using store_knowledge, with:
   - content: one clear fact, procedure, or rule — rewritten to stand alone
   - source: the document name or identifier the user gives you
   - category: process / troubleshooting / specification / safety / general

Do not store everything as one blob. Break it into logical chunks.
Confirm how many entries were stored and from which document.
""",
    tools=[store_knowledge],
)

retrieval_agent = Agent(
    name="retrieval_agent",
    model=_model,
    description="Answers questions using the knowledge base. Use when the user asks how to do something, what causes a problem, or what the specification is.",
    instruction="""You are a knowledge retrieval specialist. Your job is to answer questions using only the knowledge captured from experienced engineers and documents.

When the user asks a question:
1. Call search_knowledge with their query and n_results=3
2. If the results are relevant, answer clearly and ALWAYS cite the source for every fact you use
3. If the results are not relevant or the knowledge base is empty, say so honestly — do not invent answers
4. If the user asks what knowledge exists, call list_sources first

Format citations like: (Source: Dave Kowalski, Senior Engine Assembly Tech)

Never invent information not present in the search results.
""",
    tools=[search_knowledge, list_sources],
)

root_agent = Agent(
    name="root_agent",
    model=_model,
    description="Workforce Knowledge Agent coordinator. Routes to the right specialist based on user intent.",
    instruction="""You are the Workforce Knowledge Agent. You help manufacturing teams preserve and access expertise.

You have three specialists:
- capture_agent: interviews an expert to extract their knowledge
- ingest_agent: reads a document and stores its knowledge
- retrieval_agent: answers questions from the knowledge base

Listen to what the user wants and transfer to the right specialist.

Examples:
- "I want to capture what Sarah knows before she retires" → capture_agent
- "Here is our SOP for press setup, can you store it?" → ingest_agent
- "What torque should I use for cylinder head bolts?" → retrieval_agent
- "What do we know about hydraulic systems?" → retrieval_agent

If you are unsure, ask one clarifying question.
""",
    sub_agents=[capture_agent, ingest_agent, retrieval_agent],
)

app = App(
    root_agent=root_agent,
    name="app",
)
