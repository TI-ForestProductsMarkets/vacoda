from functools import wraps


def require_file(path_getter, hint: str):
    """
    Decorator that checks a file exists before calling the decorated function.

    Parameters
    ----------
    path_getter : callable or Path
        A callable returning a Path, or a Path directly.
    hint : str
        A helpful message shown in the error about where to look.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            path = path_getter() if callable(path_getter) else path_getter
            if not path.exists():
                raise FileNotFoundError(
                    f"File not found at: {path}\n{hint}"
                )
            return func(*args, **kwargs)
        return wrapper
    return decorator
