---
layout: page
permalink: /notes/
title: notes
description: Course notes and lecture recaps.
nav: true
nav_order: 4
---

<div class="notes">
  {% for course in site.data.notes %}
    <a class="note-card-link" href="{{ course.permalink | relative_url }}">
      <div class="card">
        <div class="card-body">
          <h2 class="card-title">{{ course.title }}</h2>
          <p class="card-text">{{ course.description }}</p>
          <p class="note-meta">
            {{ course.notes | size }} lecture note{% if course.notes.size != 1 %}s{% endif %}
          </p>
        </div>
      </div>
    </a>
  {% endfor %}
</div>
