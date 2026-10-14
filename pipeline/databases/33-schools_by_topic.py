#!/usr/bin/env python3
"""Module that contains a function to return schools having a specific topic."""


def schools_by_topic(mongo_collection, topic):
    """Returns the list of school having a specific topic.
    
    Args:
        mongo_collection: The pymongo collection object.
        topic (string): The topic searched.
        
    Returns:
        list: List of school documents matching the topic.
    """
    return list(mongo_collection.find({ "topics": topic }))
