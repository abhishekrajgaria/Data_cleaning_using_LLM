def get_hosp_brief_prompt(hospitalname, city, state):
    return f""" Your Tasks it to provide a brief about the Hospital, containing only facts. Also categorize them among ['Acute Care Hospitals' 'Acute Care - VA Medical Center''Critical Access Hospitals' 'Childrens'] and does the hospital provide emergency services or not.
    
    One example is provided, understand the structure of the Brief to provide brief to final hospital.
    
    Example:
    
    Hospital: ALLAHAN EYE FOUNDATION HOSPITAL, BIRMINGHAM, AL
    
    Brief: Callahan Eye Foundation Hospital in Birmingham, AL, is an acute care hospital specializing in ophthalmology. It is part of the UAB Health System and is located on the campus of the University of Alabama at Birmingham. The hospital provides a range of eye care services, including specialized treatments and surgeries for various eye conditions. Additionally, it offers emergency services to address urgent ophthalmic needs.
    
    Your Tasks: 
    Provide the brief of the given hospital
    
    Hospital: {hospitalname}, {city}, {state}
    
    Brief:
    """


def get_hosp_qa_prompt(dependency_name, dependency_value, target_name):
    return f""" Your task is to create a concise question given dependency name, value and the target name. The question should be a single sentence. Few examples are given below, understand them and follow the same style to generate questions.

    Examples:

    Dependency_name = providernumber, Dependency_value = 10018 and Target_name = hospitalname
    Question: Identify the hospital name for the providernumber 10018

    Dependency_name = hospitalname, Dependency_value = CALLAHAN EYE FOUNDATION HOSPITAL and Target_name = hospitaltype
    Question: What is the hospital type of CALLAHAN EYE FOUNDATION HOSPITAL?

    Dependency_name = providernumber, Dependency_value = 10018 and Target_name = city
    Question: What is the city associated with provider number 10018?

    Dependency_name = measurecode, Dependency_value = SCIP-CARD-2 and Target_name = condition
    Question: What is the condition associated with measure code SCIP-CARD-2?
    
    Dependency_name = {dependency_name}, Dependency_value = {dependency_value} and Target_name = {target_name}
    Question:
    """
