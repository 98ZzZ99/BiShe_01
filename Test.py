import os, dotenv, pathlib; ef=pathlib.Path('.env'); print('env?',ef.exists(),ef.resolve()); dotenv.load_dotenv(); print('KEY =', os.getenv('NGC_API_KEY'))

