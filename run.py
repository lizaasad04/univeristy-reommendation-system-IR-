# # Assuming your class file is called main.py and is working:
# from main import AustralianUniversitiesIR
#
# ir_system = AustralianUniversitiesIR("aus_uni.xlsx")
#
# if ir_system.load_data() and ir_system.merge_datasets():
#     query = input("Enter your course or university search query: ")
#     results = ir_system.search(query, top_n=5)
# else:
#     print("System initialization failed.")
#
#
import nltk
nltk.download('punkt_tab')
nltk.download('wordnet')

from main import AustralianUniversitiesIR

ir_system = AustralianUniversitiesIR("aus_uni.xlsx")

# Check if cached data exists
if not ir_system.load_cached_data():
    print("No cached data found. Loading from Excel and processing...")
    if ir_system.load_data() and ir_system.merge_datasets():
        ir_system.save_processed_data()
    else:
        print("System initialization failed.")
        exit()

# Now, you're ready to search without reprocessing everything
query = input("Enter your course or university search query: ",value='',key='query_input')

results = ir_system.search(query, top_n=5)