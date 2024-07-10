# Electricity and Wifi Usage

A significant portion of homes in Accra, Ghana, rely on prepaid electricity meters whereby power is cut off when the prepaid amount runs out. Unfortunately, there was no easy way to track one's electricity usage and automate the process of topping up. In my first few days at the apartment, I ran out and also wanted to investigate how much the AC in the unit contributed to the power usage.

In this project, I recorded (in a Google Sheet) regular meter readings to predict my electricity usage and send reminders to my Google Calendar to top up the meter in order to avoid unexpected power outages. My WiFi was also prepaid, and I did the same.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Authors and Acknowledgements](#authors-and-acknowledgements)

## Installation

Install the necessary packages using pip:

pip install gspread googleapiclient google.oauth2 pandas numpy matplotlib seaborn scikit-learn scipy

## Usage

Examples of how to use this project:

This project can be helpful for people new to pandas who want to learn preprocessing data, performing regression analysis, and implementing classification algorithms.

## Contributing

Guidelines for contributing to the project:

1. Fork the repository.
2. Create a new branch (git checkout -b feature-branch).
3. Commit your changes (git commit -am 'Add some feature').
4. Push to the branch (git push origin feature-branch).
5. Create a new Pull Request.

I interact with the Google API, but it would be interesting to see how the predictions can be used for other purposes.

## License
This project does not have a specific license. Anyone can use whatever they want from this project.

## Authors and Acknowledgements
Felix K. Amankona-Diawuo - Initial work - felixkad-lab
Acknowledgements to ChatGPT for providing helpful assistance during development.