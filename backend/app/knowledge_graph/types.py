from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class GraphEntityLabel(str, Enum):
    RESEARCH_PAPER = "ResearchPaper"
    AUTHOR = "Author"
    CONCEPT = "Concept"
    TOPIC = "Topic"
    PROJECT = "Project"
    RESEARCH_QUESTION = "ResearchQuestion"
    EVIDENCE = "Evidence"
    DOCUMENT = "Document"


class GraphRelationshipType(str, Enum):
    AUTHORED_BY = "AUTHORED_BY"
    MENTIONS = "MENTIONS"
    RELATED_TO = "RELATED_TO"
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    DERIVED_FROM = "DERIVED_FROM"
    USED_IN_PROJECT = "USED_IN_PROJECT"


class GraphEntity(BaseModel):
    label: GraphEntityLabel
    entity_id: str = Field(min_length=1)
    properties: dict[str, Any] = Field(default_factory=dict)


class GraphRelationship(BaseModel):
    relationship_type: GraphRelationshipType
    from_label: GraphEntityLabel
    from_id: str
    to_label: GraphEntityLabel
    to_id: str
    properties: dict[str, Any] = Field(default_factory=dict)


class GraphEntityView(BaseModel):
    label: GraphEntityLabel
    entity_id: str
    properties: dict[str, Any] = Field(default_factory=dict)
