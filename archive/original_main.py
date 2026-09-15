import PySimpleGUI as sg
import face_recognition
import cv2
import numpy as np
import random
import pickle

#Pickle File names:
pickleFaces = 'pickleFaces2.pk'
pickleNames = 'pickleNames2.pk'
picklePatientsDict = "picklePatientsDict2.pk"
picklePatientsList = "picklePatientsList2.pk"

#Person Class:
class Person:
    def __init__(self, name, age, dob, email, phone, address, city, state, zipCode, insurance, id):
        self.name = name
        self.age = age
        self.dob = dob
        self.email = email
        self.phone = phone
        self.address = address
        self.city = city
        self.state = state
        self.zip = zipCode
        if insurance == "":
            self.insured = False
            self.insurance = "Uninsured"
        else:
            self.insured = True
            self.insurance = insurance
        self.patientId = id
        self.doctorVisit = ""
        self.data = f"User ID: {self.patientId}\nName: {self.name}\nAge: {self.age}\nDate of Birth: {self.dob}\nEmail: {self.email}\nPhone Number: {self.phone}\nAddress: {self.address}, {self.city}, {self.state} {self.zip}\nInsurance: {self.insurance}\nAllergies:"
    
    def __str__(self):
        return self.name
    
    def getFaceEncoding(self):
        self.image = face_recognition.load_image_file(f"{self.patientId}.png")
        self.face_encoding = face_recognition.face_encodings(self.image)[0]
        return self.face_encoding

#Random Variables lol:
y = 0
#Camera Object:
cap = cv2.VideoCapture(1)
#List of all US States:
states = [ 'AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA',
           'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME',
           'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM',
           'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX',
           'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY']
#Patient Dictionary and List:
with open(picklePatientsDict, 'rb') as fi:
    patients = pickle.load(fi)

with open(picklePatientsList, 'rb') as fi:
    patientsList = pickle.load(fi)

#Doctor Visit List
visit = ["<1 Year", "1-2 Years", "3-4 Years", "4+ Years"]

# Access pickle (database) files:
with open(pickleFaces, 'rb') as fi:
    known_face_encodings = pickle.load(fi)

with open(pickleNames, 'rb') as fi:
    known_face_names = pickle.load(fi)

def get_image():
    retval, im = cap.read()
    return im

def face_cam():
    # Initialize some variables - used later
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    while True:
        # Grab a single frame of video
        ret, frame = cap.read()

        # Only process every other frame of video to save time
        if process_this_frame:
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]
        
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.4)
                name = "Unknown Person"

                # # If a match was found in known_face_encodings, just use the first one.
                # if True in matches:
                #     first_match_index = matches.index(True)
                #     name = known_face_names[first_match_index]

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    #print(best_match_index)
                face_names.append(name)
                #Create a temporary person to acess data from list
                tempPerson = patientsList[best_match_index]
        process_this_frame = not process_this_frame


        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 136, 0), 2)

            # Draw a label with a name and info below the face IF not unknown
            font = cv2.FONT_HERSHEY_SIMPLEX
            if name == "Unknown Person":
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (255, 136, 0), cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), font, .6, (255, 255, 255), 1)
            else:
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom + 45), (255, 136, 0), cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), font, .6, (255, 255, 255), 1)
                cv2.putText(frame, f"Age: {tempPerson.age}", (left + 6, bottom + 13), font, .5, (255, 255, 255), 1)
                cv2.putText(frame, f"DOB: {tempPerson.dob}", (left + 6, bottom + 30), font, .5, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow("Live Facial Recognition", frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release webcams
    cv2.destroyAllWindows()

#Set theme to STEP colors
sg.theme("Reddit")

#Home Layout:
home = [ 
    #logo and Title:
    [sg.Push(), sg.Image("STEP logo.png", pad=((0,0),(0,10))), sg.Push()],
    [sg.Push(), sg.Text("Medical Database and Facial Scanner", font = 'Bahnschrift 35'), sg.Push()],
    #Buffer
    [sg.Text(size=(2,2))],
    #Buttons:
    [sg.Push(), sg.Button("Upload Patient to Database", size=(25), font="Bahnschrift 25", pad=(0,10)), sg.Push()], 
    [sg.Push(), sg.Button("Live Facial Recognition", size=(25), font="Bahnschrift 25", pad=(0,10)), sg.Push()], 
    [sg.Push(), sg.Button("Access Database", size=(25), font="Bahnschrift 25", pad=(0,20)), sg.Push()],
]
#Terms and Conditions:
terms = [
    [sg.Text("By agreeing to these terms, I consent for this program to use my face and medical information \nprovided for documentation purposes. I understand this information will be saved and\nused in the STEP Science Fair Live Demo.", font= "Bahnschrift 20")],
    [sg.Button("Disagree", size=(10), font="Bahnschrift 15"), sg.Button("Agree", size=(10), font="Bahnschrift 15")]
]
#Uploading Layout:
upload = [
    [sg.Push(), sg.Text("Please take a clear photo.", font = "Bahnschrift 30"), sg.Push()],
    [sg.Text(size=(2,2))],
    [sg.Image(filename='', key='image')],
    [sg.Text(size=(2,1))],
    [sg.Push(), sg.Button('Cancel', size=(10, 1), font='Bahnschrift 15'), sg.Button("Capture", size=(10,1), font="Bahnschrift 15"), sg.Push()]
]

#Medical Form:
form = [
    #ID Number
    [sg.Text("", font="Bahnschrift 20", key="idText")],
    [sg.Text("Please enter the following information", font="Bahnschrift 25", justification="center")],
    #Name
    [sg.Text("Name:", font="Bahnschrift 17"), sg.Input("", size=(30), key="nameInput", font="Bahnschrift 15", background_color="white"), sg.Push()],
    #Age
    [sg.Text("Age:", font="Bahnschrift 17"), sg.Input("", size=(3), key="ageInput", font="Bahnschrift 15", background_color="white"), sg.Push()],
    #DOB
    [sg.Text("Date of Birth (MM/DD/YYYY):", font="Bahnschrift 17"), sg.Input("", size=(11), key="dobInput", font="Bahnschrift 15", background_color="white"), sg.Push()],
    #Email
    [sg.Text("Email:", font="Bahnschrift 17"), sg.Input("", size=(30), key="emailInput", font="Bahnschrift 15", background_color="white"), sg.Push()],
    #Phone #
    [sg.Text("Phone Number:", font="Bahnschrift 17"), sg.Input("", size=(15), key="phoneInput", font="Bahnschrift 15", background_color="white"), sg.Push()],
    [sg.Text("Address:", font="Bahnschrift 17"), sg.Input("", size=(30), key="addressInput", font="Bahnschrift 15", background_color="white"), sg.Push()],
    [sg.Text("City:", font="Bahnschrift 17"), sg.Input("", size=(20), key="cityInput", font="Bahnschrift 15", background_color="white"), sg.Push()],
    [sg.Text("State:", font="Bahnschrift 17"), sg.Listbox(values=states, select_mode="LISTBOX_SELECT_MODE_SINGLE", size=(4, 5), font="Bahnschrift 17", background_color="white", sbar_trough_color="white", key="stateInput"), sg.Push()],
    [sg.Text("Zip Code:", font="Bahnschrift 17"), sg.Input("", size=(5), key="zipInput", font="Bahnschrift 15", background_color="white"), sg.Push()],
    #Insurance Radios: 1, 2, 3
    [sg.Radio('I am uninsured', "insurance", default=True, font="Bahnschrift 10"), sg.Radio('I have public health insurance', "insurance", font="Bahnschrift 10"), sg.Radio('I have private health insurance', "insurance", font="Bahnschrift 10"), sg.Push()],
    [sg.Text("Insurance Provider:", font="Bahnschrift 17", text_color="light gray", key="insuranceText"), sg.Input("", size=(20), key="insuranceInput", font="Bahnschrift 15", background_color="white", disabled=True, disabled_readonly_background_color="light gray"), sg.Push()],
    [sg.Push() ,sg.Button('Submit', size=(10, 1), font='Bahnschrift 15'), sg.Push()]
]

#Allergies and Medical Problems:
allergies = [
    [sg.Text("Input any known allergies. Click 'Add' to add as many as you require.\nOnce finished or if you have none then simply click 'Done'", font="Bahnschrift 17")],
    [sg.Input("", size=(20), key="allergyInput", font="Bahnschrift 15", background_color="white", disabled_readonly_background_color="light gray"), sg.Button("Add", font='Bahnschrift 15', key="Add1"), sg.Button("Done", font='Bahnschrift 15', key="Done1")],
    [sg.Text(size=(2,4))],
    [sg.Text("Please do the same for any known medical complications you have.", font="Bahnschrift 17")],
    [sg.Input("", size=(20), key="problemInput", font="Bahnschrift 15", background_color="white", disabled_readonly_background_color="light gray"), sg.Button("Add", font='Bahnschrift 15', key="Add2"), sg.Button("Done", font='Bahnschrift 15', key="Done2")],
    [sg.Text(size=(2,4))],
    [sg.Text("When was the last time you have received any type of medical care or treatment\n(This includes routine checkups, ER visits, specialist appointments, etc.)", font="Bahnschrift 17"), sg.Listbox(values=visit, select_mode="LISTBOX_SELECT_MODE_SINGLE", size=(10, 4), font="Bahnschrift 17", background_color="white", sbar_trough_color="white", key="visitInput"), sg.Button("Confirm", font="Bahnschrift 17", key="Confirm1")]
]

#Database Photo Upload:
databaseUpload = [
    [sg.Push(), sg.Text("Take a photo to search for a patient in the database", font = "Bahnschrift 20"), sg.Push()],
    [sg.Push(),sg.Image(filename='', key='image2'), sg.Push()],
    [sg.Push(), sg.Button('Cancel', size=(10, 1), font='Bahnschrift 15', key="Cancel2"), sg.Button("Capture", size=(10,1), font="Bahnschrift 15", key="Capture2"), sg.Push()],
    [sg.Text("Or instead, use the patient's User ID:", font="Bahnschrift 15"), sg.Input("", size=(8), key="userIdInput", font="Bahnschrift 15", background_color="white"), sg.Button("Submit", font = "Bahnschrift 15", key = "Submit3"), sg.Push()]
]

#Database Search:
databaseSearch = [
    [sg.Push(), sg.Text("", font = "Bahnschrift 17", key="searchText"), sg.Push()],
    [sg.Push(), sg.Text("", font = "Bahnschrift 17", key="searchText2"), sg.Push()],
    [sg.Text(size=(2,2))],
    [sg.Push(), sg.Multiline(size=(50,15), disabled=True, font = "Bahnschrift 17", key="displayData", no_scrollbar=True), sg.Push()],
    [sg.Push(), sg.Button("Return Home", key="Return Home2", font = "Bahnschrift 17"), sg.Push()]
]

#Search Failed:
searchFailed = [
    [sg.Push(), sg.Text("User not found, please return to home page.", font = "Bahnschrift 17"), sg.Push()],
    [sg.Button("Return Home", font = "Bahnschrift 17")]
]

#Main layout
layout = [
    [sg.Sizer(215,0), sg.Column(home, visible=True, key="col1", size=(3840, 2160)), sg.Push()],
    [sg.Column(terms, visible=False, key="terms", size=(3840,2160)), sg.Push()], 
    [sg.Sizer(305,0), sg.Column(upload, visible=False, key="col2", size=(3840,2160))],
    [sg.Sizer(300,0), sg.Column(form, visible=False, key="col3", size=(3840,2160), element_justification='c'), sg.Push()],
    [sg.Column(allergies, visible=False, key="allergy", size=(3840,2160)), sg.Push()],
    [sg.Sizer(305,0), sg.Column(databaseUpload, visible=False, key="databaseUpload", size=(3840,2160))],
    [sg.Sizer(305,0), sg.Column(databaseSearch, visible=False, key="databaseSearch", size=(3840,2160))],
    [sg.Column(searchFailed, visible=False, key="searchFailed", size=(3840,2160))]
]


#Creating the main window:
window = sg.Window("STEP Medical Database", layout, location=(-10,0))
recording = False
recording2 = False
#Event Loop!
while True:
    event, values = window.read(timeout = 20)
    #Closing the program:
    if event == sg.WIN_CLOSED:
        break
    #Webcam button:
    if event == "Live Facial Recognition":
        face_cam()
    
    if event == "Upload Patient to Database":
        window["allergyInput"].update(disabled=False)
        window["problemInput"].update(disabled=False)
        window["col1"].update(visible=False)
        window["terms"].update(visible=True)
    #Uploading Patients:
    if event == "Agree":
        recording = True
        window["terms"].update(visible=False)
        window["col2"].update(visible=True)
    if event == "Disagree":
        window["terms"].update(visible=False)
        window["col1"].update(visible=True)
    #Webcam preview:
    if recording:
        ret, frame = cap.read()
        imgbytes = cv2.imencode('.png', frame)[1].tobytes()  # ditto
        window['image'].update(data=imgbytes)
    #Cancelling Upload
    if event == 'Cancel':
        recording = False
        img = np.full((480, 640), 255)
        # this is faster, shorter and needs less includes
        imgbytes = cv2.imencode('.png', img)[1].tobytes()
        window['image'].update(data=imgbytes)
        window["col1"].update(visible=True)
        window["col2"].update(visible=False)
    
    #Taking Photo
    if event == "Capture":
        for i in range(30):
            temp = get_image()
        camera_capture = get_image()
        userId = random.randint(1000000, 9999999)
        cv2.imwrite(f"{userId}.png", camera_capture)
        window["idText"].update(f"Your user id is: {userId}. Please note this down")
        window["col2"].update(visible=False)
        window["col3"].update(visible=True)

    if values[1] != True:
        window["insuranceInput"].update(disabled=False)
        window["insuranceText"].update(text_color="black")
    else:
        window["insuranceInput"].update("")
        window["insuranceInput"].update(disabled=True)
        window["insuranceText"].update(text_color="light gray")

    if event == "Submit":
        persone = Person(name=values["nameInput"], age=values["ageInput"], dob=values["dobInput"], email=values["emailInput"], phone=values["phoneInput"], address=values["addressInput"], city=values["cityInput"], state=values["stateInput"], zipCode=values["zipInput"], insurance=values["insuranceInput"], id = userId)
        known_face_encodings.append(persone.getFaceEncoding())
        known_face_names.append(persone.name)
        window["nameInput"].update("")
        window["ageInput"].update("")
        window["dobInput"].update("")
        window["emailInput"].update("")
        window["phoneInput"].update("")
        window["addressInput"].update("")
        window["cityInput"].update("")
        window["zipInput"].update("")
        window["insuranceInput"].update("")
        window["col3"].update(visible=False)
        window["allergy"].update(visible=True)

    if event == "Add1":
        allergy = values["allergyInput"]
        persone.data += f" {allergy},"
        window["allergyInput"].update("")
        y += 1
        
    if event == "Done1":
        if y == 0:
            persone.data += " None"
        y = 0
        window["allergyInput"].update(disabled=True)
        window["allergyInput"].update("")
        persone.data += "\nMedical Complications:"
    
    if event == "Add2":
        problems = values["problemInput"]
        persone.data += f" {problems},"
        window["problemInput"].update("")
        y += 1
        
    if event == "Done2":
        if y == 0:
            persone.data += " None"
        window["problemInput"].update(disabled=True)
        window["problemInput"].update("")

    if event == "Confirm1":
        persone.doctorVisit = values["visitInput"]
        persone.data += f"\nMost Recent Medical Visit (As of 2022): {persone.doctorVisit}"
        window["allergy"].update(visible=False)
        window["col1"].update(visible=True)
        patients[f"{userId}"] = persone
        patientsList.append(persone)
    
    if event == "Access Database":
        recording2 = True
        window["col1"].update(visible=False)
        window["databaseUpload"].update(visible=True)
    
    if recording2:
        ret, frame = cap.read()
        imgbytes = cv2.imencode('.png', frame)[1].tobytes()  # ditto
        window['image2'].update(data=imgbytes)
    #Cancelling Upload
    if event == 'Cancel2':
        recording2 = False
        img = np.full((480, 640), 255)
        # this is faster, shorter and needs less includes
        imgbytes = cv2.imencode('.png', img)[1].tobytes()
        window['image2'].update(data=imgbytes)
        window["databaseUpload"].update(visible=False)
        window["col1"].update(visible=True)
    
    #Taking Photo
    if event == "Capture2":
        for i in range(30):
            temp = get_image()
        camera_capture = get_image()
        unknownId = random.randint(1000000, 9999999)
        cv2.imwrite(f"unknown{unknownId}.png", camera_capture)
        unknown_image = face_recognition.load_image_file(f"unknown{unknownId}.png")
        unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]
        results = face_recognition.compare_faces(known_face_encodings, unknown_face_encoding, tolerance=0.5)
        if True in results:
            foundUser = results.index(True)
            foundPatient = patientsList[foundUser]
            window["searchText"].update(f"User {foundPatient.patientId} found!")
            window["searchText2"].update(f"Medical Information for {foundPatient.name}:")
            window["displayData"].update("")
            window["displayData"].print(foundPatient.data)
            window["databaseUpload"].update(visible=False)
            window["databaseSearch"].update(visible=True)
        else:
            window["databaseUpload"].update(visible=False)
            window["searchFailed"].update(visible=True)
    
    if event == "Submit3":
        foundId = values["userIdInput"]
        try:
            foundPatient = patients[foundId]
            window["searchText"].update(f"User {foundPatient.patientId} found!")
            window["searchText2"].update(f"Medical Information for {foundPatient.name}:")
            window["displayData"].update("")
            window["displayData"].print(foundPatient.data)
            window["databaseUpload"].update(visible=False)
            window["databaseSearch"].update(visible=True)
        except:
            window["databaseUpload"].update(visible=False)
            window["searchFailed"].update(visible=True)
            
    if event == "Return Home":
        window["userIdInput"].update("")
        window["searchFailed"].update(visible=False)
        window["col1"].update(visible=True)
    
    if event == "Return Home2":
        window["userIdInput"].update("")
        window["databaseSearch"].update(visible=False)
        window["col1"].update(visible=True)

#Pickle Imports:
with open(pickleFaces, 'wb') as fi:
    # dump your data into the file
    pickle.dump(known_face_encodings, fi)

with open(pickleNames, 'wb') as fi:
    # dump your data into the file
    pickle.dump(known_face_names, fi)

with open(picklePatientsDict, 'wb') as fi:
    # dump your data into the file
    pickle.dump(patients, fi)

with open(picklePatientsList, 'wb') as fi:
    # dump your data into the file
    pickle.dump(patientsList, fi)
