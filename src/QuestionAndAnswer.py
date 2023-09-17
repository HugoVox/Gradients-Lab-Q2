import gradio as gr
import qa_utils as utils

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
wiki, conn, cursor = utils.import_dataset()
print("Database connected!\n")

print("Loading retrieval model...")
qar_model = utils.retrieve_model()
print("Done!\n")

print("Loading Q&A Sentence to sentence model...")
path = "models\eli5_bart_model"
qas_model, qas_tokenizer = utils.generate_model(model_id=path, backbone = "yjernite/bart_eli5")
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
