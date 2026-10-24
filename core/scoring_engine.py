def ai_score(prob):

    score = round(prob * 100)

    if score >= 50:
        rating = "BUY"
    elif score >= 20:
        rating = "WATCH"
    else:
        rating = "AVOID"

    return score, rating
