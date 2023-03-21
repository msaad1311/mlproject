import sys


def get_error_message(error, error_details: sys):
    _, _, error_tbe = error_details.exc_info()
    file_name = error_tbe.tb_frame.f_code.co_filename
    error_message = f"Error occured in the {file_name} at {error_tbe.tb_frame.f_lineno} with {str(error)}"
    return error_message


class MyException(Exception):
    def __init__(self, error_message, error_details):
        super().__init__(error_message)
        self.error_message = get_error_message(
            error=error_message, error_details=error_details
        )

    def __str__(self) -> str:
        return self.error_message
