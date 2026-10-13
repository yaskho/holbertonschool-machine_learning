-- Creates a stored procedure ComputeAverageWeightedScoreForUser that computes and stores the average weighted score for a student
DELIMITER //

CREATE PROCEDURE ComputeAverageWeightedScoreForUser(IN user_id INT)
DETERMINISTIC
BEGIN
    DECLARE weighted_avg FLOAT;
    
    SELECT SUM(c.score * p.weight) / SUM(p.weight) INTO weighted_avg
    FROM corrections c
    JOIN projects p ON c.project_id = p.id
    WHERE c.user_id = user_id;
    
    IF weighted_avg IS NULL THEN
        SET weighted_avg = 0;
    END IF;

    UPDATE users
    SET average_score = weighted_avg
    WHERE id = user_id;
END //

DELIMITER ;
