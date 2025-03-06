setTimeout(() => {
  document.getElementById('loading').classList.add('hidden');
  document.getElementById('main-content').classList.remove('hidden');
}, 0);

const canvas = document.getElementById('backgroundCanvas');
const context = canvas.getContext("2d");

function resizeCanvas() {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
}
resizeCanvas();
window.onresize = resizeCanvas;

var dotCount = 200;
var dots = [];

for (var i = 0; i < dotCount; i++) {
  dots.push(new dot());
}

function render() {
  context.fillStyle = "#141333";
  context.fillRect(0, 0, canvas.width, canvas.height);
  for (var i = 0; i < dotCount; i++) {
      dots[i].draw();
      dots[i].move();
  }
  requestAnimationFrame(render);
}

function dot() {
  this.rad_x = 2 * Math.random() * window.innerWidth / 2 + 1;
  this.rad_y = 1.2 * Math.random() * window.innerHeight / 2 + 1;
  this.alpha = Math.random() * 360 + 1;
  this.speed = Math.random() * 100 < 50 ? 1 : -1;
  this.speed *= 0.01;
  this.size = Math.random() * 5 + 1;
  this.c = Math.random() > 0.5 ? "#2720be" : "#00d869";
}

dot.prototype.draw = function() {
  var dx = window.innerWidth / 2 + this.rad_x * Math.cos(this.alpha / 180 * Math.PI);
  var dy = window.innerHeight / 2 + this.rad_y * Math.sin(this.alpha / 180 * Math.PI);
  context.fillStyle = this.c;
  context.fillRect(dx, dy, this.size, this.size);
};

dot.prototype.move = function() {
  this.alpha += this.speed;
};

render();



let selectedFiles = [];

function validateFiles() {
  const fileInput = document.getElementById('fileInput');
  selectedFiles = Array.from(fileInput.files);

  if (selectedFiles.length < 2 || selectedFiles.length > 10) {
      alert('Пожалуйста, загрузите от 2 до 10 фотографий');
      return;
  }
}

async function uploadFiles() {
  if (selectedFiles.length < 2 || selectedFiles.length > 10) {
      alert('Пожалуйста, загрузите от 2 до 10 фотографий');
      return;
  }

  const formData = new FormData();
  selectedFiles.forEach((file) => {
      formData.append('files', file);
  });

  document.getElementById('loading-spinner').classList.remove('hidden');

  try {
      const response = await fetch('/verify', {
          method: 'POST',
          body: formData,
      });

      const result = await response.json();
      document.getElementById('loading-spinner').classList.add('hidden');
      console.log(result);

      if (response.ok) {
          if (selectedFiles.length === 2) {
              const photo1 = selectedFiles[0];
              const photo2 = selectedFiles[1];

              document.getElementById('comparisonResult').innerHTML = `
      <div class="person-row">
        <div class="same-person">
          <img src="${URL.createObjectURL(photo1)}" alt="Фото 1" class="uploaded-photo">
        </div>
        <div class="same-person">
          <img src="${URL.createObjectURL(photo2)}" alt="Фото 2" class="uploaded-photo">
        </div>
      </div>
    `;

              console.log(result)
              if (result.verification_matrix && result.verification_matrix[0] !== undefined) {
                  const verificationResult = result.verification_matrix[0][1];
                  const resultText = verificationResult >= 0.37 ?
                      `Верифицировано: ${verificationResult.toFixed(2)}` :
                      `Не верифицировано: ${verificationResult.toFixed(2)}`;
                  const colorResult = verificationResult >= 0.37 ? 'button-verified' : 'button-not-verified';

                  document.getElementById('comparisonResult').innerHTML += `
        <div class="verification-result ${colorResult}">
          ${resultText}
        </div>
      `;
              } else {
                  alert('Ошибка в данных с сервера');
              }
          }

          if (selectedFiles.length >= 3) {
              console.log('dfd')
              generateResultTable(result.verification_matrix, result.file_names);
          }

      } else {
          alert(`Ошибка: ${result.error}`);
      }
  } catch (error) {
      console.error('Ошибка:', error);
      document.getElementById('loading-spinner').classList.add('hidden');
      alert('Произошла ошибка при отправке фотографий');
  }
}

function generateResultTable(matrix, fileNames) {
  const tableBlock = document.getElementById('resultTableBlock');
  tableBlock.style.display = 'block';

  const table = document.getElementById('resultsTable');
  table.innerHTML = '';

  const headerRow = document.createElement('tr');
  const emptyHeader = document.createElement('th');
  headerRow.appendChild(emptyHeader);

  fileNames.forEach((fileName, index) => {
      const th = document.createElement('th');
      const img = document.createElement('img');
      img.src = URL.createObjectURL(selectedFiles[index]);
      img.alt = fileName;
      img.classList.add('table-image');
      th.appendChild(img);
      headerRow.appendChild(th);
  });
  table.appendChild(headerRow);

  fileNames.forEach((fileName, i) => {
      const row = document.createElement('tr');
      const rowHeader = document.createElement('td');
      const img = document.createElement('img');
      img.src = URL.createObjectURL(selectedFiles[i]);
      img.alt = fileName;
      img.classList.add('table-image');
      rowHeader.appendChild(img);
      row.appendChild(rowHeader);

      if (matrix && matrix[i]) {
          matrix[i].forEach((cosineSimilarity, j) => {
              const td = document.createElement('td');

              if (i === j) {
                  td.textContent = 'NULL';
                  td.classList.remove('similarity-high', 'similarity-low');
              } else {
                  td.textContent = cosineSimilarity.toFixed(2);

                  if (cosineSimilarity >= 0.37) {
                      td.classList.add('similarity-high');
                  } else {
                      td.classList.add('similarity-low');
                  }
              }

              row.appendChild(td);
          });
      } else {
          alert('Некорректные данные для формирования таблицы');
      }
      table.appendChild(row);
  });
}