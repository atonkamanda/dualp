from termcolor import colored

def log_info(message):
    print(colored(message, 'green'))

def log_error(message):
    print(colored(message, 'red'))

# Example usage:
log_info("This is an informational log message.")
log_error("This is an error log message.")
