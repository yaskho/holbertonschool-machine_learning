#!/usr/bin/env python3
"""Module that contains a function to return all students sorted by average score."""


def top_students(mongo_collection):
    """Returns all students sorted by average score.
    
    Args:
        mongo_collection: The pymongo collection object.
        
    Returns:
        list: List of student documents with their averageScore, ordered.
    """
    pipeline = [
        {
            "$addFields": {
                "averageScore": {"$avg": "$topics.score"}
            }
        },
        {
            "$sort": {"averageScore": -1}
        }
    ]
    return list(mongo_collection.aggregate(pipeline))
