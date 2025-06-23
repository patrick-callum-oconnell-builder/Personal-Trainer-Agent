import pytest
from backend.knowledge_graph import KnowledgeGraph

def test_entity_creation():
    """Test creating and retrieving entities."""
    kg = KnowledgeGraph(kg_path=None)
    kg.patterns = []
    kg._add_entity("PERSON", "John", {"age": 30})
    entity = kg.get_entity("John")
    assert entity is not None
    assert entity.type == "PERSON"
    assert entity.attributes["age"] == 30

def test_relation_creation():
    """Test creating and retrieving relations."""
    kg = KnowledgeGraph(kg_path=None)
    kg.patterns = []
    kg._add_entity("PERSON", "John", {})
    kg._add_entity("FOOD", "Pizza", {})
    kg._add_relation("John", "Pizza", "LIKES")

    relations = kg.get_relations("John")
    assert len(relations) == 1
    assert relations[0].type == "LIKES"
    assert relations[0].target == "Pizza"

def test_prompt_parsing():
    """Test parsing a prompt to build the graph."""
    kg = KnowledgeGraph(kg_path=None)
    prompt = "My name is Alice. I am 30. I like swimming and hiking."
    kg.parse_prompt(prompt)

    # Check root person
    assert kg.root_person == "Alice"

    # Check entities
    assert "Alice" in kg.entity_map
    assert "30" in kg.entity_map
    assert "swimming" in kg.entity_map
    assert "hiking" in kg.entity_map
    assert kg.entity_map["swimming"].type == "PREFERENCE"

    # Check relations
    relations = kg.get_relations("Alice")
    assert len(relations) == 4 # IS_NAMED, HAS_AGE, LIKES (swimming), LIKES (hiking)
    
    # Let's check for the LIKES relationship more robustly
    likes_relations = [r for r in relations if r.type == "LIKES"]
    liked_items = {r.target for r in likes_relations}
    assert "swimming" in liked_items
    assert "hiking" in liked_items


def test_persistence():
    """Test saving and loading the graph from a file."""
    # This test requires a file path
    import os
    test_path = "test_kg.json"
    kg = KnowledgeGraph(kg_path=test_path)
    kg.patterns = []
    kg._add_entity("PERSON", "Bob", {})
    kg.save_to_file()

    new_kg = KnowledgeGraph(kg_path=test_path)
    entity = new_kg.get_entity("Bob")
    assert entity is not None
    assert entity.type == "PERSON"
    
    # Clean up the test file
    if os.path.exists(test_path):
        os.remove(test_path)


def test_query_method():
    """Test the query method for finding entities and relations."""
    kg = KnowledgeGraph(kg_path=None)
    kg.patterns = []
    
    # Add test data
    kg._add_entity("PERSON", "John", {})
    kg._add_entity("FOOD", "Pizza", {})
    kg._add_entity("FOOD", "Burger", {})
    kg._add_relation("John", "Pizza", "LIKES")
    kg._add_relation("John", "Burger", "LIKES")

    # Query all FOOD entities
    results = kg.query(entity_type="FOOD")
    assert len(results) == 2

    # Query all LIKES relations
    results = kg.query(relation_type="LIKES")
    assert len(results) == 2

def test_pattern_matching():
    """Test the pattern matching functionality."""
    kg = KnowledgeGraph(kg_path=None)
    kg.patterns = []
    # Add a custom pattern
    kg.add_pattern(r"enjoys ([\w\s]+)", "ACTIVITY", "ENJOYS")
    kg._add_entity("PERSON", "John", {}) # Manually add John

    # Test pattern matching
    sentence = "John enjoys swimming"
    kg._extract_entities(sentence)
    kg._extract_relationships(sentence)

    assert "swimming" in kg.entity_map
    relations = kg.get_relations("John")
    assert any(r.type == "ENJOYS" and r.target == "swimming" for r in relations)

def test_relationship_indicators():
    """Test the relationship indicator functionality."""
    kg = KnowledgeGraph(kg_path=None)
    kg.patterns = []
    kg._add_entity("PERSON", "John", {}) # Manually add John
    kg._add_entity("ACTIVITY", "swimming", {}) # Manually add swimming

    # Add a custom relationship indicator
    kg.add_relationship_indicator("enjoys", "ENJOYS")

    # Test relationship extraction
    sentence = "John enjoys swimming"
    kg._extract_relationships(sentence)

    relations = kg.get_relations("John")
    assert any(r.type == "ENJOYS" and r.target == "swimming" for r in relations) 