---
layout: page
permalink: /notes/machine-learning/
title: Machine Learning
description: Lecture notes and recaps for the Machine Learning course.
nav: false
---

<div class="notes">
  {% assign course = site.data.notes | where: "title", "Machine Learning" | first %}
  {% for note in course.notes %}
    <a class="note-card-link" href="{{ note.permalink | relative_url }}">
      <div class="card">
        <div class="card-body">
          <h2 class="card-title">{{ note.title }}</h2>
          <p class="card-text">{{ note.description }}</p>
          <p class="note-meta">
            Date: {{ note.date | date: "%B %-d, %Y" }} |
            Estimated Reading Time: {{ note.reading_time }}
          </p>
        </div>
      </div>
    </a>
  {% endfor %}
</div>
