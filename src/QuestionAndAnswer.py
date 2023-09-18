import gradio as gr
import qa_utils as utils
import os

#! For Running on local machine, uncomment the following codes
# path = r'D:\Gradients\Gradients-Lab-Q2'   #* Change this path to your local path
# os.chdir(path)
# print(f'Directory is changed to {os.getcwd()}.')


def get_answer(question):
    global wiki, data, qar_model, qas_model, conn, cursor
    context = utils.query(question, qar_model, conn, cursor, wiki)
    question_doc = "question: {} context: {}".format(question, context)
    answer = utils.generate_answer(question_doc, qas_model, qas_tokenizer)[0]
    return answer

print("Checking database files...")
utils.check_database_folder()
print("Done!\n")

print("Initializing database...")
utils.init_database()
print("Done!\n")

print("Connecting to database...")
wiki, conn, cursor = utils.connect_to_database()
print("Database connected!\n")

print("Getting available device...")
device = utils.get_device()

print("Loading retrieval model...")
qar_model = utils.retrieve_model(device)
print("Done!\n")

print("Loading Q&A Sentence to sentence model...")
path = "models\eli5_bart_model"
qas_model, qas_tokenizer = utils.generate_model(model_id=path, 
                                                backbone = "yjernite/bart_eli5",
                                                device=device)
print("Done!\n")

print("Initializing Gradio Inteface...")
demo = gr.Interface(fn=get_answer, 
                    inputs="textbox", 
                    outputs="textbox",
                    title="Question and Answer",
                    description="Question and Answer by Team 2\nMembers:\n  - Vo Duy Khoa\n - Nguyen Minh Dang\n  - Nguyen Thach Ha Anh")

print("Waiting for running...")
try:
    demo.launch(inbrowser=True,
                share=False)
except KeyboardInterrupt:
    demo.close()
    utils.close_database()
except Exception as e:
    print(e)
    demo.close()
    utils.close_database()
