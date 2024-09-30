module load compiler/intel/2019u5/intelpython3
>>> import ssl
# Bypass SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('punkt')
