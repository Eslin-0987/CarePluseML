/**
 * CarePulse Health Assessment & ML Prediction Interactivity
 */

document.addEventListener('DOMContentLoaded', () => {
  const searchInput = document.getElementById('symptomSearchInput');
  const categoryButtons = document.querySelectorAll('.category-filter-btn');
  const symptomBlocks = document.querySelectorAll('.symptom-category-block');
  const symptomCards = document.querySelectorAll('.symptom-checkbox-card');
  const selectedTray = document.getElementById('selectedSymptomsTray');
  const selectedChipsList = document.getElementById('selectedChipsList');
  const selectedCountBadge = document.getElementById('selectedCountBadge');
  const reviewCount = document.getElementById('reviewSelectedCount');
  const reviewPreview = document.getElementById('reviewSymptomsPreview');
  const reviewModel = document.getElementById('reviewSelectedModel');
  const assessmentForm = document.getElementById('assessmentForm');
  const submitBtn = document.getElementById('predictSubmitBtn');
  const processingOverlay = document.getElementById('processingOverlay');

  // Track selected symptom IDs
  const selectedSymptoms = new Map(); // id -> label

  // Initialize from pre-checked checkboxes if any
  symptomCards.forEach(card => {
    const checkbox = card.querySelector('input[type="checkbox"]');
    if (checkbox.checked) {
      const symId = checkbox.value;
      const label = card.querySelector('.symptom-name').textContent.trim();
      selectedSymptoms.set(symId, label);
      card.classList.add('checked');
    }

    // Toggle on card click
    card.addEventListener('click', (e) => {
      if (e.target !== checkbox) {
        checkbox.checked = !checkbox.checked;
      }
      handleCheckboxChange(checkbox, card);
    });

    checkbox.addEventListener('change', () => {
      handleCheckboxChange(checkbox, card);
    });
  });

  function handleCheckboxChange(checkbox, card) {
    const symId = checkbox.value;
    const label = card.querySelector('.symptom-name').textContent.trim();

    if (checkbox.checked) {
      selectedSymptoms.set(symId, label);
      card.classList.add('checked');
    } else {
      selectedSymptoms.delete(symId);
      card.classList.remove('checked');
    }
    updateSelectedTray();
  }

  function updateSelectedTray() {
    selectedChipsList.innerHTML = '';
    const count = selectedSymptoms.size;
    selectedCountBadge.textContent = `${count} selected`;
    if (reviewCount) reviewCount.textContent = count;

    if (count === 0) {
      selectedChipsList.innerHTML = '<span style="font-size: 13px; color: var(--text-muted); font-style: italic;">No symptoms selected yet. Search or check items below.</span>';
      if (reviewPreview) reviewPreview.textContent = 'None';
    } else {
      const labels = [];
      selectedSymptoms.forEach((label, id) => {
        labels.push(label);
        const chip = document.createElement('div');
        chip.className = 'selected-chip';
        chip.innerHTML = `
          <span>${label}</span>
          <button type="button" class="selected-chip-remove" data-id="${id}" aria-label="Remove ${label}">&times;</button>
        `;
        selectedChipsList.appendChild(chip);
      });

      if (reviewPreview) {
        reviewPreview.textContent = labels.slice(0, 4).join(', ') + (labels.length > 4 ? ` (+${labels.length - 4} more)` : '');
      }
    }

    // Attach remove listeners
    document.querySelectorAll('.selected-chip-remove').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const idToRemove = btn.getAttribute('data-id');
        const targetCheckbox = document.querySelector(`input[name="symptoms"][value="${idToRemove}"]`);
        if (targetCheckbox) {
          targetCheckbox.checked = false;
          const card = targetCheckbox.closest('.symptom-checkbox-card');
          if (card) card.classList.remove('checked');
        }
        selectedSymptoms.delete(idToRemove);
        updateSelectedTray();
      });
    });
  }

  // Initial tray update
  updateSelectedTray();

  // 2. Real-Time Search Filter
  if (searchInput) {
    searchInput.addEventListener('input', () => {
      const query = searchInput.value.toLowerCase().trim();
      
      // Reset category filter to All when actively searching
      categoryButtons.forEach(btn => btn.classList.remove('active'));
      const allBtn = document.querySelector('.category-filter-btn[data-category="All"]');
      if (allBtn) allBtn.classList.add('active');

      symptomBlocks.forEach(block => {
        let hasVisibleInBlock = false;
        const cards = block.querySelectorAll('.symptom-checkbox-card');
        
        cards.forEach(card => {
          const text = card.querySelector('.symptom-name').textContent.toLowerCase();
          const match = text.includes(query);
          card.style.display = match ? 'flex' : 'none';
          if (match) hasVisibleInBlock = true;
        });

        block.style.display = hasVisibleInBlock ? 'block' : 'none';
      });
    });
  }

  // 3. Category Filter Buttons
  categoryButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      categoryButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const selectedCat = btn.getAttribute('data-category');

      if (searchInput) searchInput.value = '';

      symptomBlocks.forEach(block => {
        const blockCat = block.getAttribute('data-category');
        const showBlock = (selectedCat === 'All' || blockCat === selectedCat);
        block.style.display = showBlock ? 'block' : 'none';

        if (showBlock) {
          block.querySelectorAll('.symptom-checkbox-card').forEach(c => {
            c.style.display = 'flex';
          });
        }
      });
    });
  });

  // 4. Model Selection Radio Cards
  const modelCards = document.querySelectorAll('.model-card');
  modelCards.forEach(card => {
    card.addEventListener('click', () => {
      modelCards.forEach(c => c.classList.remove('selected'));
      card.classList.add('selected');
      const radio = card.querySelector('input[type="radio"]');
      radio.checked = true;
      if (reviewModel) {
        reviewModel.textContent = card.querySelector('.model-title').textContent.trim();
      }
    });
  });

  // 5. Form Submission with Validation and Processing State
  if (assessmentForm) {
    assessmentForm.addEventListener('submit', (e) => {
      if (selectedSymptoms.size === 0) {
        e.preventDefault();
        alert('Please select at least one symptom to proceed with the assessment.');
        const symptomsSection = document.getElementById('stepSymptoms');
        if (symptomsSection) symptomsSection.scrollIntoView({ behavior: 'smooth' });
        return;
      }

      // Disable button and show processing overlay
      if (submitBtn) {
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<span>Processing Model...</span>';
      }
      if (processingOverlay) {
        processingOverlay.style.display = 'flex';
      }
    });
  }
});
