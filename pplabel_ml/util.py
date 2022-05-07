import connexion


def abort(detail, status, title=""):
    raise connexion.exceptions.ProblemException(
        detail=detail, title=title, status=status, headers={"message": detail}
    )

