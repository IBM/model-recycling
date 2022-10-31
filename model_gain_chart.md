---
layout: default
---
<h1 id="header">Model gains</h1>
<h2 id="avg">Model gains</h2>
<canvas id="myChart"></canvas>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script>
    const urlParams = new URLSearchParams(window.location.search);
    document.getElementById('header').innerHTML = urlParams.get("model_name") + ' model gains'
    document.getElementById('avg').innerHTML = 'Model avg gain: ' + urlParams.get("avg")
    urlParams.delete("base_name");
    urlParams.delete("model_name");
    urlParams.delete("avg");
    urlParams.delete("mnli_lp");

    var items = Array.from(urlParams.keys()).map(function (key) {
        return [key, urlParams.get(key)];
    });
    items.sort(function (first, second) {
        return second[1] - first[1];
    });

    const data = {
        labels: items.map(e => e[0]),
        datasets: [{
            backgroundColor: 'rgb(252, 201, 86)',
            borderColor: 'rgb(254, 86, 50)',
            data: items.map(e => e[1]),
        }]
    };
    const config = {
        type: 'bar',
        data: data,
        options: {
            scales:{
              x:{
                  title:{
                    display: true,
                    text: 'gain',
                  },
              },
            },
            indexAxis: 'y',
            elements: {
                bar: {
                    borderWidth: 2,
                }
            },
            responsive: true,
            plugins: {
                legend: {
                    display: false,
                    position: 'right',
                },
                title: {
                    display: true,
                    text: ''
                }
            }
        },
    };
    const myChart = new Chart(
        document.getElementById('myChart'),
        config
    );
</script>
