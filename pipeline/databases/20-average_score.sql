-- Creates a stored procedure ComputeAverageScoreForUser that computes and stores the average score for a student
DELIMITER //
CREATE PROCEDURE ComputeAverageScoreForUser(
    IN user_id INT
)
BEGIN
    DECLARE avg_score FLOAT;

    -- Compute the average score from corrections for the given user
    SELECT AVG(score) INTO avg_score FROM corrections WHERE corrections.user_id = user_id;

    -- Update the average_score field in the users table
    UPDATE users SET average_score = avg_score WHERE id = user_id;
END;
//
DELIMITER ;
