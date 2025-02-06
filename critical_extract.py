import streamlit as st
import pandas as pd
import pyaudio
import queue
import os
import google.generativeai as genai
from google.cloud import speech
from dotenv import load_dotenv
import speech_recognition as sr
import json


load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "accounts_key.json"
gemini_api_key = os.getenv('GEMINI_API_KEY')


st.subheader("Doctor Feedback")
if 'text' not in st.session_state:
    st.session_state['text'] = "Listening..."
    st.session_state['run'] = False
    st.session_state['final_transcription'] = ""
    st.session_state['sentiment_analysis'] = {}
    st.session_state['critical_analysis']= ""

# Define language options
language_options = {
    "English (US)": "en-US",
    "Hindi (India)": "hi-IN",
    "Marathi (India)": "mr-IN"
}

# Add dropdown for language selection
selected_language = st.sidebar.selectbox("Select Preferred Language", list(language_options.keys()))
selected_language_code = language_options[selected_language]


# Initialize PyAudio
# audio_queue = queue.Queue()
# audio_interface = pyaudio.PyAudio()

# stream = audio_interface.open(
#     format=pyaudio.paInt16,
#     channels=1,
#     rate=16000,
#     input=True,
#     frames_per_buffer=3200,
#     stream_callback=lambda in_data, frame_count, time_info, status: (
#         audio_queue.put(in_data),
#         pyaudio.paContinue,
#     )
# )

# # Google Speech-to-Text Client
# client = speech.SpeechClient()
# config = speech.RecognitionConfig(
#     encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
#     sample_rate_hertz=16000,
#     language_code=selected_language_code,
# )
# streaming_config = speech.StreamingRecognitionConfig(
#     config=config,
#     interim_results=True
# )

# Real-Time Transcription Function
# def transcribe_stream():
#     audio_generator = (audio_queue.get() for _ in iter(int, 1))
#     requests = (
#         speech.StreamingRecognizeRequest(audio_content=chunk)
#         for chunk in audio_generator
#     )

#     responses = client.streaming_recognize(streaming_config, requests)

#     for response in responses:
#         if not st.session_state['run']:
#             break
#         if response.results:
#             result = response.results[0]
#             if result.is_final:
#                 transcription = result.alternatives[0].transcript
#                 st.session_state['final_transcription'] += transcription + " "
#                 st.session_state['text'] = transcription
#                 st.write(st.session_state['text'])


def listen():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        while st.session_state["run"]:  # Check session state to allow stopping
            try:
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source)
                result = recognizer.recognize_google(audio, language=selected_language_code)
                
                # Update session state similar to PyAudio logic
                st.session_state["final_transcription"] += result + " "
                st.session_state["text"] = result
                st.write(st.session_state["text"])
                
            except sr.UnknownValueError:
                st.write("Sorry, I could not understand the audio.")
            except sr.RequestError as e:
                st.write(f"Could not request results; {e}")
            except Exception as e:
                st.write(f"An error occurred: {e}")


doctor_sentiment_analysis = """Identify and summarize the doctor's sentiment towards the pharma company, its products, or the industry in general. Capture both explicit and implicit sentiments, and categorize them accordingly. Ensure the output is structured in a tabular format with the columns: Doctor Sentiment, Reason, and Context. With no additional text before or after the table.  Summarize the reason concisely. The response must be in english

Output Format:

| Doctor Sentiment | Reason                          | Context                        |
|-----------------|---------------------------------|--------------------------------|
| [Positive/Neutral/Negative] | [Brief explanation],  max 15 words | [Mentioned product, service, or issue],  max 15 words |"""


location_based_performance = """Identify and summarize any mentions of location, city, or region in the transcript. If the pharma company's performance in that region is mentioned, include an appropriate tag. Additionally, note any relevant observations on market demand or competition. Ensure the output is structured in a tabular format with the columns: Region, Performance, and Observations.With no additional text before or after the table. Summarize observations in a few words.The response must be in english

Output Format:

| Region    | Performance | Observations                         |
|-----------|------------|--------------------------------------|
| [Region]  | [Good/Neutral/Poor] | [Brief insights],  max 15 words  |"""

competitor_performance_analysis = """Identify any mentions of competing pharma companies in the transcript. Note which company is performing better or is being preferred in a specific region or by the doctor. Ensure the output is structured in a tabular format with the columns: Competitor Mentioned, Region, and Competitive Edge.  With no additional text before or after the table. Summarize the competitive edge in a few words.The response must be in english

Output Format:

| Competitor Mentioned | Region         | Competitive Edge                          |
|----------------------|---------------|-------------------------------------------|
| [Company Name]      | [Region]       | [Brief reason for preference],  max 15 words  |"""


hardware_equipment_in_clinic = """Extract any mentions of new digital or hardware equipment observed in the clinic. Note what was installed or being used and its relevance to medical practice. Ensure the output is structured in a tabular format with the columns: New Equipment, Purpose, and Competitor. With no additional text before or after the table. Summarize the purpose concisely.The response must be in english

Output Format:

| New Equipment     | Purpose                     | Competitor       |
|------------------|----------------------------|------------------|
| [Equipment name] | [Short description],  max 15 words     | [Competitor Name] |"""


old_equipment_needing_replacement = """Identify any mentions of outdated or old equipment in the clinic that may require replacement. Provide a recommendation if available. Ensure the output is structured in a tabular format with the columns: Old Equipment Noticed and Condition.  With no additional text before or after the table. Summarize the condition concisely.The response must be in english

Output Format:

| Old Equipment Noticed | Condition                      |
|----------------------|--------------------------------|
| [Equipment name]     | [Functional/Outdated/Needs Replacement] |"""


educational_materials_used_by_doctor = """Extract details on the type of educational materials used by the doctor (e.g., brochures, online courses, training manuals). If a specific name is mentioned, include it; otherwise, omit that field. Ensure the output is structured in a tabular format with the columns: Education Material Type and Specific Name.  With no additional text before or after the table. Summarize concisely.The response must be in english

Output Format:

| Education Material Type | Specific Name          |
|------------------------|------------------------|
| [Brochure/Manual/Online Course, etc.] | [Name] |"""

research_journals_subscribed_to = """List the medical journals the doctor is subscribed to or follows for research papers. If any pharmaceutical companies are mentioned as publishers of these papers, include them. If no specific journals or companies are mentioned, omit the respective field. Ensure the output is structured in a tabular format with the columns: Journals Subscribed and Published by Pharma Companies.  With no additional text before or after the table.The response must be in english

Output Format:

| Journals Subscribed     | Published by Pharma Companies |
|------------------------|--------------------------------|
| [Journal names]        | [Company names]               |"""


conferences_attended_by_doctor = """Identify and summarize any mentions of conferences the doctor has attended or plans to attend. If a specific conference is not mentioned, provide a general note on their conference participation. Ensure the output is structured in a tabular format with the columns: Conference Name and Frequency of Attendance. With no additional text before or after the table.The response must be in english

Output Format:

| Conference Name        | Frequency of Attendance       |
|------------------------|-------------------------------|
| [Conference name]      | [Regular/Occasional/First Time] |
"""


most_discussed_medicine_or_product = """Identify which medicine or product generated the most interest during the conversation. Note if the doctor had a positive or neutral response toward it. Ensure the output is structured in a tabular format with the columns: Product Name and Interest Level. With no additional text before or after the table.The response must be in english

Output Format:

| Product Name          | Interest Level  |
|----------------------|----------------|
| [Product Name]      | [High/Moderate/Low] |
"""


digital_tools_software_in_use = """Extract any mentions of digital tools, software, or applications used by the doctor or clinic. Note their purpose and whether they are proprietary or third-party solutions. Ensure the output is structured in a tabular format with the columns: Tool/Software Name, Purpose, and Observations.  With no additional text before or after the table.The response must be in english

Output Format:

| Tool/Software Name   | Purpose                  | Observations                        |
|----------------------|-------------------------|-------------------------------------|
| [Name]              | [EHR/Telemedicine/Patient Management, etc.],  max 15 words | [Ease of use, adoption level, any noted gaps],  max 15 words |
"""


local_healthcare_infrastructure_observations = """Identify and summarize any observations about the local healthcare infrastructure (e.g., availability of hospitals, pharmacies, diagnostic centers, accessibility issues, medical supply chain efficiency). Ensure the output is structured in a tabular format with the columns: Infrastructure Observed, Condition, and Challenges.  With no additional text before or after the table.The response must be in english

Output Format:

| Infrastructure Observed | Condition          | Challenges                    |
|-------------------------|--------------------|-------------------------------|
| [Hospitals, Pharmacies, Labs, etc.] | [Well-Developed/Average/Lacking] | [Short description],  max 15 words         |
"""


emerging_treatment_patterns = """Extract insights on emerging treatment preferences or shifts in prescribing habits. Identify any specific drugs, therapies, or new approaches being discussed. Ensure the output is structured in a tabular format with the columns: Treatment Trend, Medications Mentioned, and Doctor's Sentiment.  With no additional text before or after the table.The response must be in english

Output Format:

| Treatment Trend                    | Medications Mentioned      | Doctor's Sentiment |
|-------------------------------------|----------------------------|--------------------|
| [E.g., Preference for biologics, increase in telemedicine consultations] | [Drug names]              | [Positive/Neutral/Skeptical] |
"""


unmet_needs_gaps_identified = """Identify any unmet medical needs or gaps in patient care, treatment options, or product availability as mentioned by the doctor. Note if any specific product or solution could address these gaps. Ensure the output is structured in a tabular format with the columns: Unmet Need/Gaps and Potential Solutions.  With no additional text before or after the table.The response must be in english

Output Format:

| Unmet Need/Gaps                                | Potential Solutions             |
|------------------------------------------------|---------------------------------|
| [E.g., Lack of access to a specific drug, need for affordable alternatives],  max 15 words | [E.g., Generic alternatives, better distribution],  max 15 words|
"""

need_to_bring_goodies_gifts_for_doctors = """Identify if the transcript mentions a preference or expectation for promotional materials, samples, or gifts for a specific doctor or clinic. Note any preferences expressed by the doctor. Ensure the output is structured in a tabular format with the columns: Doctor/Clinic Mentioned, Type of Goodies/Gifts Expected, and Reason.  With no additional text before or after the table.The response must be in english

Output Format:

| Doctor/Clinic Mentioned  | Type of Goodies/Gifts Expected | Reason                              |
|--------------------------|-------------------------------|-------------------------------------|
| [Name]                   | [Samples, Educational Material, Branded Items] | [E.g., Doctor prefers patient education kits],  max 15 words |
"""


price_sensitivity_feedback = """Extract any mentions of price-related concerns regarding medicines, treatments, or medical devices. Identify if the doctor or clinic has expressed concerns about affordability, patient reluctance due to cost, or the need for cheaper alternatives. Ensure the output is structured in a tabular format with the columns: Product/Service Mentioned, Price Sensitivity Level, Doctor's Feedback, and Suggested Solutions.  With no additional text before or after the table.The response must be in english

Output Format:

| Product/Service Mentioned | Price Sensitivity Level | Doctor’s Feedback                          | Suggested Solutions             |
|---------------------------|-------------------------|--------------------------------------------|---------------------------------|
| [Medicine/Device Name]     | [High/Moderate/Low]     | [E.g., "Patients are switching to generics due to high costs."],  max 15 words | [E.g., Discounts, alternative pricing models],  max 15 words|
"""

patient_volume_trends = """Extract insights on patient footfall trends. Identify whether the doctor has observed an increase, decrease, or stable patient volume. If a change is noted, extract possible reasons (e.g., seasonal factors, economic issues, pandemics, or new competition). Ensure the output is structured in a tabular format with the columns: Patient Volume Trend, Reason for Change, and Specialty Impacted.  With no additional text before or after the table.The response must be in english

Output Format:

| Patient Volume Trend | Reason for Change            | Specialty Impacted      |
|----------------------|------------------------------|-------------------------|
| [Increasing/Decreasing/Stable] | [E.g., "More seasonal flu cases", "Economic downturn", "New clinic nearby"],  max 15 words | [E.g., Cardiology, Pediatrics, General Medicine] |
"""


disease_prevelance_changes = """Identify any mentions of changing disease patterns in the doctor’s practice. Extract whether any specific conditions are increasing or decreasing in frequency and possible contributing factors (e.g., lifestyle changes, outbreaks, new treatments). Ensure the output is structured in a tabular format with the columns: Disease/Condition Noted, Prevalence Change, and Possible Reasons.  With no additional text before or after the table.The response must be in english

Output Format:

| Disease/Condition Noted | Prevalence Change    | Possible Reasons                          |
|-------------------------|----------------------|-------------------------------------------|
| [E.g., Diabetes, Hypertension] | [Increasing/Decreasing/Stable] | [E.g., "More sedentary lifestyles", "New vaccination program", "Seasonal outbreak"] ,  max 15 words|
"""


treatment_protocol_changes = """Extract any mentions of modifications in standard treatment approaches. Identify if doctors are switching medications, changing dosage guidelines, adopting new therapies, or discontinuing older treatments. Ensure the output is structured in a tabular format with the columns: Treatment Change Noted, Reason for Change, and Impact on Pharma Business.  With no additional text before or after the table. The response must be in english

Output Format:

| Treatment Change Noted                                | Reason for Change                          | Impact on Pharma Business                      |
|-------------------------------------------------------|--------------------------------------------|------------------------------------------------|
| [E.g., "Switching from Drug A to Drug B for better efficacy"] | [E.g., "Better patient outcomes", "Regulatory update", "New clinical guidelines", max 15 words] | [E.g., "Shift in preference may reduce sales of Drug A"] , max 15 words|
"""


product_sentiment_analysis = """Identify and summarize the sentiment towards the company's products and competitor products. Capture both explicit and implicit sentiments, and categorize them accordingly. If no specific sentiment, reason, or context is mentioned, omit the respective field from the output. Ensure the output is structured in a tabular format with the columns: Product Name, Sentiment Score, Category, and Additional Note.  With no additional text before or after the table.The response must be in english

Output Format:

| Product Name      | Sentiment Score | Category    | Additional Note                                    |
|-------------------|-----------------|-------------|---------------------------------------------------|
| [Product name]    | [Quantitative value] | [Competitor/Company] | [Brief explanation of what they liked or disliked,  max 15 words] |
"""


# Perform sentiment analysis for three prompts
def perform_sentiment_analysis(transcription: str, prompt:str) -> dict:
    """Generate responses for the same transcription using three defined prompts."""
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    full_pr= f"{prompt}: {transcription}"
    answer = model.generate_content(full_pr)
    prompt_list = answer.candidates[0].content.parts[0].text.strip()
    prompt_list= prompt_list.replace("```json", "").replace("```", "").strip()
    dict_prompt= json.loads(prompt_list)
    responses = {}

    variable_map = {
    "doctor_sentiment_analysis": doctor_sentiment_analysis,
    "location_based_performance": location_based_performance,
    "competitor_performance_analysis": competitor_performance_analysis,
    "hardware_equipment_in_clinic": hardware_equipment_in_clinic,
    "old_equipment_needing_replacement": old_equipment_needing_replacement,
    "educational_materials_used_by_doctor": educational_materials_used_by_doctor,
    "research_journals_subscribed_to": research_journals_subscribed_to,
    "conferences_attended_by_doctor": conferences_attended_by_doctor,
    "most_discussed_medicine_or_product": most_discussed_medicine_or_product,
    "digital_tools_software_in_use": digital_tools_software_in_use,
    "local_healthcare_infrastructure_observations": local_healthcare_infrastructure_observations,
    "emerging_treatment_patterns": emerging_treatment_patterns,
    "unmet_needs_gaps_identified": unmet_needs_gaps_identified,
    "need_to_bring_goodies_gifts_for_doctors": need_to_bring_goodies_gifts_for_doctors,
    "price_sensitivity_feedback": price_sensitivity_feedback,
    "patient_volume_trends": patient_volume_trends,
    "disease_prevalence_changes": disease_prevelance_changes,
    "treatment_protocol_changes": treatment_protocol_changes,
    "product_sentiment_analysis": product_sentiment_analysis
}
    prompts = {key: variable_map[value] + f"\n\n{transcription}" for key, value in dict_prompt.items()}

    for key, full_prompt in prompts.items():
        try:
            response = model.generate_content(full_prompt)
            output_text = response.candidates[0].content.parts[0].text.strip()
            responses[key] = output_text
        except (IndexError, AttributeError) as e:
            responses[key] = f"Error generating response: {e}"

    return responses

# Start/Stop Buttons
def start_listening():
    st.session_state['run'] = True
    st.session_state['text'] = "Listening..."
    st.session_state['final_transcription'] = ""
    st.session_state['sentiment_analysis'] = {}
    st.session_state['critical_analysis']= ""

def stop_listening():
    st.session_state['run'] = False

# Download Transcription
def download_transcription():
    if st.session_state['final_transcription']:
        st.download_button(
            label="Download Transcription",
            data=st.session_state['final_transcription'],
            file_name="transcription.txt",
            mime="text/plain"
        )

st.title('Real-Time Transcription and Sentiment Analysis App')
col1, col2 = st.columns(2)

col1.button('Start', on_click=start_listening)
col1.button('Stop', on_click=stop_listening)





# Start/Stop Transcription
if st.session_state['run']:
    with st.spinner("Listening... Speak into the microphone."):
        listen()


# Prompts

classify_prompts= """ Analyze the given transcript in detail and evaluate which of the following parameters apply based on the context. Identify only the parameters that provide real-time market signals. Extract actionable insights that help business heads understand market trends, competition, and emerging needs.

Evaluate the transcript based on the following parameters:

    doctor_sentiment_analysis
    location_based_performance
    competitor_performance_analysis
    hardware_equipment_in_clinic
    old_equipment_needing_replacement
    educational_materials_used_by_doctor
    research_journals_subscribed_to
    conferences_attended_by_doctor
    most_discussed_medicine_or_product
    digital_tools_software_in_use
    local_healthcare_infrastructure_observations
    emerging_treatment_patterns
    unmet_needs_gaps_identified
    need_to_bring_goodies_gifts_for_doctors
    price_sensitivity_feedback
    patient_volume_trends
    disease_prevalence_changes
    treatment_protocol_changes
    product_sentiment_analysis

Return only a dictionary where the keys are formatted in title case and the values are in snake_case. Do not include any additional text or explanations.

Example Output Format:
{
"Doctor Sentiment Analysis": "doctor_sentiment_analysis",
"Most Discussed Medicine/Product": "most_discussed_medicine_or_product",
"Unmet Needs/Gaps Identified": "unmet_needs_gaps_identified",
"Research Journals Subscribed To": "research_journals_subscribed_to"
}

Only include parameters that are supported by the context of the transcript.
"""

def extract_critical(transcription: str, prompt:str) -> str:
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    full_pr= f"{prompt}: {transcription}"
    answer = model.generate_content(full_pr)
    prompt_list = answer.candidates[0].content.parts[0].text.strip()

    return prompt_list




extract_data= """Prompt:
You are given a transcript generated from an audio recording of a medical representative's (MR) visit to a doctor. Your task is to extract structured information based on predefined sections while ensuring accuracy in medicine name identification.

Rules for Extraction:
1. Section-1: Visit Details
Identify the type of visit: Independent or Joint Visit
Extract the timeslot: Morning, Evening, or Night
List other doctors the MR met at the clinic
2. Section-2: Discussed Products
Extract medicine names discussed from the transcript with the doctor. Ensure only name of medicines are extracted that the mr claims to have discussed with the doctor.
Use the provided medicine list to correct misrecognized medicine names from the transcript:  [
"Abevia-N 100mg/600mg Tablet",
"Abiways-M Tablet SR",
"Acnestar 2.5% Benzoyl Peroxide Soap 
"Alkanam Syrup",
"Alograce Moisturising Cream with Aloe Vera 
"Amlogift 5mg Tablet",
"Amlogift AT 5mg/50mg Tablet",
"Amlokind 10mg Tablet",
"Amlokind-H Tablet",
"Amrobrut Cream",
"Antokind D 30mg/40mg Capsule SR",
"Aptimust Oral Drops",
"Arnisac 100 Tablet",
"Asthakind Oral Suspension Sugar Free Mango",
"Asthakind Tablet",
"Asthakind-P Drops",
"Azikind XL 200mg Oral Suspension",
"Aztrohigh 2000mg Injection",
"Aztrohigh 500mg Injection",
"Baloforce 100mg Tablet",
"Bandy-Plus Suspension",
"Barikind 4 Tablet",
"Betaglim M DS 2mg/500mg Tablet",
"Betakind Gargle Mint",
"Birtybrom Pet Powder",
"Bisoheart-T 5mg/40mg Tablet",
"Body Fuel Powder",
"Brutacef 100mg Dry Syrup",
"Brutacef AZ 200 mg/500 mg Tablet",
"Brutacef O Tablet",
"Brutapenem 500mg Injection",
"Cadistar Syrup",
"Calapure-A Lotion with Aloe Vera 
"Calapure-A Lotion with Aloe Vera 
"Calcimust Pet Liquid",
"Caldikind -P Delicious Mango Oral Suspension",
"Caldikind -P New Delicious Mango Oral Suspension",
"Caldikind Capsule",
"Cancrinase 3750IU Injection",
"Candiforce 100 Capsule",
"Candiforce 200 Capsule",
"Candiforce SB 130mg Capsule",
"Cardimol-Plus 10 Tablet SR",
"Cefablast CV 500mg/125mg Tablet",
"Cefaclass 50mg Dry Syrup",
"Cefakind XP 250mg/31.25mg Injection",
"Cefastar-CV Dry Syrup",
"Ceftiforce SB 2000mg/1000mg Injection",
"Ceftiforce SB 2000mg/1000mg Pet Injection",
"Chekfall 10% Spray/Solution"
]
If a medicine is not found in the provided list, keep it as is
3. Section-3: Sample Distribution
Identify if samples were provided
For each medicine name mentioned in the transcript, extract the following details:
Available quantity: 120 (This is fixed, and will be used to calculate the pending quantity)
Total quantity: 120 (This is fixed)
Provided quantity: (Extracted directly from the transcript)
Pending quantity: (Calculated as Available quantity - Provided quantity)
Format the output in a tabular manner with the columns:
Medicine Name
Available Quantity
Total Quantity
Provided Quantity
Pending Quantity
4. Section-4: Items Given
Identify if non-medicine items were provided
For each item name mentioned in the transcript, extract the following details:
Available quantity: 10 (This is fixed, and will be used to calculate the pending quantity)
Total quantity:  10 (This is fixed)
Provided quantity: (Extracted directly from the transcript)
Pending quantity: (Calculated as Available quantity - Provided quantity)
Format the output in a tabular manner with the columns:
Item Name
Available Quantity
Total Quantity
Provided Quantity
Pending Quantity
5. Section-5: Division Market Share
Extract and display the division market share or DMU in a clear format
Examples of Extraction & Formatting:
Input Transcript:
"This was an independent visit in the evening. I met Dr. Sharma and Dr. Mehta at the clinic. I discussed Acnestar soap, Amlo gift AT tablet, and some antibiotics. I provided samples of Amrobrut cream—available 50, total 100, provided 20, pending 30. Also, I gave out promotional pens—available 200, total 500, provided 100, pending 100. The division market share for the doctor is 1000."

Expected Output:
### Section-1: Visit Details  
- **Visit Type:** Independent  
- **Timeslot:** Evening  
- **Other Doctors Met:** Dr. Sharma, Dr. Mehta  

### Section-2: Discussed Products  
- **Medicines Discussed:** Acnestar 2.5% Benzoyl Peroxide Soap, Amlogift AT 5mg/50mg Tablet  

### Section-3: Samples Provided  
| Medicine Name         | Available Quantity | Total Quantity | Provided Quantity | Pending Quantity |  
|----------------------|------------------|--------------|----------------|----------------|  
| Amrobrut Cream      | 50               | 100          | 20             | 30             |  

### Section-4: Items Provided  
| Item Name           | Available Quantity | Total Quantity | Provided Quantity | Pending Quantity |  
|---------------------|------------------|--------------|----------------|----------------|  
| Promotional Pens   | 200              | 500          | 100            | 100            |  

### Section-5: Division Market Share  
- **Division Market Share:** 1000
Instructions for the Model:
Parse the transcript carefully and extract information according to the predefined sections.
Use the provided medicine list to correct any misrecognized medicine names based on context.
Maintain proper formatting with headings, bullet points, and tables for clarity.
Handle missing data gracefully—if a section is missing in the transcript, indicate "No data available" for that section."""


# Display final transcription after stopping
if not st.session_state['run'] and st.session_state['final_transcription']:
    st.markdown("### Final Transcription")
    st.text_area("Transcription", st.session_state['final_transcription'], height=200)
    
    # Perform sentiment analysis via button
    if st.button('Perform Sentiment Analysis'):
        with st.spinner("Analyzing sentiment..."):
            st.session_state['critical_analysis'] = extract_critical(
                st.session_state['final_transcription'],
                extract_data
            )

            st.session_state['sentiment_analysis']= perform_sentiment_analysis(
                st.session_state['final_transcription'],
                classify_prompts
            )

    # Display sentiment analysis results
    if st.session_state['sentiment_analysis'] and st.session_state['critical_analysis']:
        st.markdown("### Key Insights")

        st.write(st.session_state['critical_analysis'])
        st.write("--------------------------------------------------------")
        st.markdown("### Sentiment Analysis Results")

        prompt_names = list(st.session_state['sentiment_analysis'].keys())

        # Create a DataFrame to display the tabular format
        df = pd.DataFrame(prompt_names, columns=["Parameters Used for Analysis"])

        # Display the table first
        st.dataframe(df)
        st.write("--------------------------------------------------------")
        for prompt_name, response in st.session_state['sentiment_analysis'].items():
            st.markdown(f"**{prompt_name}:**")
            st.write(response)
            st.write("--------------------------------------------------------")

# Provide option to download transcription
download_transcription()


