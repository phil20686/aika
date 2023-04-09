class Validators:
    """
    All validators should return their value[s] if valid,
    else raise a ValueError
    """

    class Integers:
        @classmethod
        def check_type(cls, value: int, name: str):
            if isinstance(value, int):
                return value
            else:
                raise ValueError(f"Parameter {name} must be an int")

        @classmethod
        def greater_than_zero(cls, value: int, name: str) -> int:
            """

            Parameters
            ----------
            value: int
            name: str
                Used in the error messages only.
            Returns
            -------
            """
            cls.check_type(value, name)
            if not (value > 0):
                raise ValueError(f"Parameter {name} must be greater than zero")
            return value
