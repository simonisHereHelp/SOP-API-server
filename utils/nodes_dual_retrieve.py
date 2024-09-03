from document import Document
from utils.generate_chain import create_generate_chain

class GraphNodes:
    def __init__(self, llm, answers_list_retriever, question_retriever, retrieval_grader, hallucination_grader, code_evaluator, question_rewriter):
        self.llm = llm
        self.retriever_answer_list = answers_list_retriever
        self.retriever_question = question_retriever
        self.retrieval_grader = retrieval_grader
        self.hallucination_grader = hallucination_grader
        self.code_evaluator = code_evaluator
        self.question_rewriter = question_rewriter
        self.generate_chain = create_generate_chain(llm)

    def retrieve(self, state):
        """
        Retrieve documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        print("---RETRIEVE (Langgraphy flow)---")
        question = state["input"]
        question_document = self.retriever_question.invoke("Extract the question from the document")
        question_combined = question+"  Here is the extracted question: "+question_document
        # Retrieval
        documents = self.retriever_answer_list.invoke(question_combined)
        # Also, retrieve the question document itself (e.g., to compare or refine the question)
        
        return {"documents": documents, "input": question_combined}


    def generate(self, state):
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        print("---GENERATE---")
        question = state["input"]
        documents = state["documents"]

        # RAG generation
        generation = self.generate_chain.invoke({"context": documents, "input": question})
        return {"documents": documents, "input": question, "generation": generation}

    def grade_documents(self, state):
        """
        Determines whether the retrieved answers list documents are relevant to the question which is defined with the attachment doc.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with only filtered relevant documents
        """
        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["input"]
        documents = state["documents"]

        # score each doc
        filtered_docs = []

        for d in documents:
            score = self.retrieval_grader.invoke({"input": question, "document": d.page_content})
            grade = score["score"]
            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT IR-RELEVANT---")
                continue

        return {"documents": filtered_docs, "input": question}

    def transform_query(self, state):
        """
        Transform the query to produce a better question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates question key with a re-phrased question
        """
        print("---TRANSFORM QUERY---")
        question = state["input"] + self.retriever_question("what is the question in the doc?")
        documents = state["documents"]

        # Re-write question
        better_question = self.question_rewriter.invoke({"input": question})
        return {"documents": documents, "input": better_question}