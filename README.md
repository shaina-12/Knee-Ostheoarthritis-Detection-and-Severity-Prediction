# Knee Ostheoarthritis Detection and Severity Prediction Using Machine Learning
This Repository is under development. It will be developed on 7th May 2023.
# Important Questions
Q1) What is the generalized objective of the project?
Ans) This project aims to build a model that will detect and grade the severity of knee osteoarthritis using deep learning techniques on knee X-ray images. These solutions can potentially enhance the lives of persons affected by this very frequent disease by helping and speeding diagnosis to deploy appropriate countermeasures.
Q2) What is the actual objective of the project?
Ans) To develop a new algorithm for the detection and severity grading of knee osteoarthritis.
Q3) What kind of features we are targeting for the detection and severity grading of Knee Osteoarthritis?
Ans) In general we are targeting the following from the x-ray images:
●	Joint space narrowing
●	Osteophyte formation
●	Subchondral sclerosis
●	Subchondral cysts
Till now whatever information we have, we are going to take joint space narrowing and osteophyte formation as two main features. Rest will be considered based on the demand of the project *,!.
* When the affected joint is examined with an X-ray, subchondral sclerosis can appear as a dense area of bone just under the cartilage in your joints, and it looks like abnormally white bone along the joint line. Magnetic resonance imaging (MRI) is also a good test for visualizing soft tissue damage. 
! SBCs can be diagnosed using an X-ray. If a cyst isn’t clear on an X-ray image, your doctor may order an MRI of the affected joint. In addition to these images, your doctor will ask about your medical history, symptoms of osteoarthritis, and risk factors. That information along with images can help your doctor correctly diagnose subchondral bone cysts.
More Information:
[1] https://www.ncbi.nlm.nih.gov/books/NBK507884/
[2] https://www.verywellhealth.com/what-is-subchondral-sclerosis-2552142#:~:text=When%20the%20affected%20joint%20is,for%20visualizing%20soft%20tissue%20damage.
[3] https://www.healthline.com/health/osteoarthritis/subchondral-bone-cyst#symptoms
[4] https://www.radiologymasterclass.co.uk/tutorials/musculoskeletal/imaging-joints-bones/osteoarthritis
Q4) On what basis we are determining whether the person is having knee osteoarthritis or not?
Ans) We are using the KL (Kellgren-Lawrence) Grading System which is a widely used grading system for the severity grading of knee osteoarthritis. 
Description of the Kellgren-Lawrence Classification System:
Based on the data presented in their original work, the KL classification is typically applied specifically within the context of knee OA. The KL classification was originally described using AP knee radiographs. Each radiograph was assigned a grade from 0 to 4, which correlated to increasing severity of OA, with Grade 0 signifying no presence of OA and Grade 4 signifying severe OA(Fig. 1). Additionally, KL provided detailed radiographic descriptions of OA. Although it is unclear from the original paper whether the radiographic descriptions were presented with the intent of demonstrating a linear disease progression of OA that begins with the formation of osteophytes and culminates in the altered shape of bone ends, other authors have criticized the KL system based on this assumption.
More Information:
[1] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4925407/#:~:text=Description%20of%20the%20Kellgren%2DLawrence%20Classification%20System&text=The%20KL%20classification%20was%20originally,OA%20%5B19%5D%20(Fig.
Q5) What kind of dataset we are using for the project and from where you are getting this dataset?
Ans) In this project:
●	In total, 9786 cropped right and left knee X-ray images with a resolution of 299x299 pixels were obtained from Mendeley Data in Chen's "Knee Osteoarthritis Severity Grading Dataset" and published paper. 
●	From 9786 X-ray images, 8270 were used in this project of which 5778 X-ray images were used for training, 826 X-ray images were used for validation and 5778 X-ray images were used for testing. 
●	This dataset was obtained from the OAI and edited by cropping just the parts of the knee joint that are divided into right and left joints.
More Information:
[1] https://data.mendeley.com/datasets/56rmx5bjcr
[2] https://nda.nih.gov/oai/
Q6) What is the methodology of the project?
Ans) 
Q7) Can we able to achieve the goals?
Ans)
Extra Links:
[1] https://monai.io/
[2] https://github.com/Project-MONAI
[3] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5336370/




