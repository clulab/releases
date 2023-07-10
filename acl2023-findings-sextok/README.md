# SexTok  
  
**Title :** It’s not Sexually Suggestive; It’s Educative | Separating Sex Education from Suggestive Content on TikTok videos

Enfa George, Mihai Surdeanu, ACL Findings 2023 [[bib]](#citation)


[Computational Language Understanding Lab](https://clulab.org/), 
University Of Arizona

**Outline**

  - [Paper](#paper)
  - [Motivation](#motivation)
  - [Data](#data)
  - [Example](#example) 
  - [Results](#results)
  - [Citation](#citation)

## Paper   
  
[Findings of the Association for Computational Linguistics: ACL 2023](https://aclanthology.org/2023.findings-acl.365/)

## Motivation

The current state of adolescent sex education in the United States is often criticized for being fragmented and inadequate, susceptible to political influence, and lacking comprehensive information. Only a small number of states require contraception education, and even fewer cover important topics like gender diversity and consent. This limited focus hampers the effectiveness of sex education programs, as highlighted by the American Academy of Pediatrics. 

<img src= "https://www.truthdig.com/wp-content/uploads/2017/06/sexed_500.jpg" width="40%"><img src="https://image.cagle.com/107727/750/107727.png" width="40%">

<sup align="centre"> Credit: Left - [Truthdig](https://www.truthdig.com/cartoons/abstinence-only-sex-ed) Right - [Cagle](https://www.cagle.com/pat-bagley/2012/03/sex-education)</sup>
</div>


Meanwhile, TikTok, a widely popular app among adolescents and youth, provides a platform for virtual sex education in a convenient, private, and inclusive space for sexual health information. However, these videos often face removal and shadow banning due to inaccuracies in community guidelines enforcement and mass reporting. Creators, especially those from marginalized communities, are disproportionately targeted by mass reporting. 

<img src = "https://helios-i.mashable.com/imagery/articles/02Fvbn4j2IGVdt4DU8Wy0gX/images-399.fit_lim.size_376x.jpg" height=300 align="centre"><img src=https://helios-i.mashable.com/imagery/articles/02Fvbn4j2IGVdt4DU8Wy0gX/images-401.fit_lim.size_376x.jpg height=300 align="centre">

<sup> Credit : [Mashable - Why is TikTok removing sex ed videos?](https://mashable.com/article/tiktok-sex-education-content-removal) </sup>

The project's goal is to develop a better system for distinguishing between sexual content and sex education, creating a dataset and establishing a baseline for future research.
  
## Data 

You can find the data file as a CSV [here](https://github.com/enfageorge/SexTok/data). The CSV contains the following information - Video Link,	Data split, Gender Expression, Label, and Notes, if any. The videos were given as URLs to avoid any potential copyright violation. In the event that any of the videos are taken down, please contact the author for a copy.

## Example 

### Illustrative example

![Example](https://github.com/enfageorge/SexTok/blob/data/docs/images/example.png)

### Text

**As Description:**
 - Educative: Video featuring a man discussing a topic while a prominent illustration of a p*n*s with pearly penile papules serves as the background.
 - Suggestive: The video shows a man holding a pumpkin over his torso while a woman enthusiastically moves her hand inside, exclaiming, "There is so much in there."

**As Transcript:**

- Educative: The average banana in the United States is about 5.5 inches long. That’s the perfect size for baking banana bread most of the time because ...
- Suggestive: You are such a good boy. Daddy’s so proud of you.

## Results 

<table>
    <tr>
        <td><b>Group</b></td>
        <td><b>Acc</b></td>
        <td colspan="3"><b>Micro</b></td>
        <td colspan="3"><b>Macro</b></td>
    </tr>
    <tr>
        <td></td>
        <td></td>
        <td>P</td>
        <td>R</td>
        <td>F1</td>
        <td>P</td>
        <td>R</td>
        <td>F1</td>
    </tr>
    <tr>
        <td>Majority</td>
        <td>0.60</td>
        <td>0.00</td>
        <td>0.00</td>
        <td>0.00</td>
        <td>0.20</td>
        <td>0.33</td>
        <td>0.25</td>
    </tr>
    <tr>
        <td>All Text</td>
        <td>0.68&pm; 0.06</td>
        <td>0.76&pm; 0.06</td>
        <td>0.50&pm; 0.06</td>
        <td>0.60&pm; 0.04</td>
        <td>0.71&pm; 0.06</td>
        <td>0.63&pm; 0.03</td>
        <td>0.64&pm; 0.04</td>
    </tr>
    <tr>
        <td>Non-empty Text</td>
        <td>0.75&pm; 0.02</td>
        <td>0.78&pm; 0.07</td>
        <td>0.54&pm; 0.02</td>
        <td>0.64&pm; 0.02</td>
        <td>0.74&pm; 0.04</td>
        <td>0.65&pm; 0.01</td>
        <td>0.68&pm; 0.00</td>
    </tr>
    <tr>
        <td>Video</td>
        <td>0.70&pm; 0.04</td>
        <td>0.61&pm; 0.11</td>
        <td>0.51&pm; 0.07</td>
        <td>0.55&pm; 0.05</td>
        <td>0.68&pm; 0.06</td>
        <td>0.57&pm; 0.07</td>
        <td>0.61&pm; 0.01</td>
    </tr>
</table>


## Citation  
 ```
@inproceedings{george-surdeanu-2023-sexually,  
    title = "It{'}s not Sexually Suggestive; It{'}s Educative | Separating Sex Education from Suggestive Content on {T}ik{T}ok videos",  
    author = "George, Enfa  and  
      Surdeanu, Mihai",  
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",  
    month = jul,  
    year = "2023",  
    address = "Toronto, Canada",  
    publisher = "Association for Computational Linguistics",  
    url = "https://aclanthology.org/2023.findings-acl.365",  
    pages = "5904--5915",  
    abstract = "We introduce SexTok, a multi-modal dataset composed of TikTok videos labeled as sexually suggestive (from the annotator{'}s point of view), sex-educational content, or neither. Such a dataset is necessary to address the challenge of distinguishing between sexually suggestive content and virtual sex education videos on TikTok. Children{'}s exposure to sexually suggestive videos has been shown to have adversarial effects on their development (Collins et al. 2017). Meanwhile, virtual sex education, especially on subjects that are more relevant to the LGBTQIA+ community, is very valuable (Mitchell et al. 2014). The platform{'}s current system removes/punishes some of both types of videos, even though they serve different purposes. Our dataset contains video URLs, and it is also audio transcribed. To validate its importance, we explore two transformer-based models for classifying the videos. Our preliminary results suggest that the task of distinguishing between these types of videos is learnable but challenging. These experiments suggest that this dataset is meaningful and invites further study on the subject.",  
}  
 ```
