#include <iostream>
#include <string>

#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>

class TCPClient
{
private:
    int socketFD;

public:
    TCPClient()
    {
        socketFD = -1;
    }

    bool connectToServer(const char *ip, int port)
    {
        socketFD = socket(AF_INET, SOCK_STREAM, 0);

        if (socketFD < 0)
            return false;

        sockaddr_in serverAddress{};

        serverAddress.sin_family = AF_INET;
        serverAddress.sin_port = htons(port);

        if (inet_pton(AF_INET, ip, &serverAddress.sin_addr) != 1)
            return false;

        if (connect(socketFD,
                    (sockaddr *)&serverAddress,
                    sizeof(serverAddress)) < 0)
        {
            return false;
        }

        return true;
    }

    bool sendPrediction(int prediction)
    {
        if (prediction < 0 || prediction > 9)
            return false;

        std::string data = std::to_string(prediction) + "\n";

        size_t totalSent = 0;

        while (totalSent < data.size())
        {
            ssize_t sent = send(
                socketFD,
                data.data() + totalSent,
                data.size() - totalSent,
                0);

            if (sent <= 0)
                return false;

            totalSent += static_cast<size_t>(sent);
        }

        return true;
    }

    ~TCPClient()
    {
        if (socketFD >= 0)
            close(socketFD);
    }
};

#include <iostream>
#include <fstream>
#include <string>
#include <thread>
#include <chrono>

// Include your TCPClient class here.

int main()
{
    TCPClient client;

    if (!client.connectToServer("192.168.2.112", 5000))
    {
        std::cerr << "Connection failed!\n";
        return 1;
    }

    std::cout << "Connected to Windows!\n";

    const std::string filename = "prediction.txt";

    std::string lastPrediction;

    while (true)
    {
        std::ifstream file(filename);

        if (file.is_open())
        {
            std::string prediction;

            if (std::getline(file, prediction))
            {
                if (!prediction.empty() && prediction != lastPrediction)
                {
                    int number = std::stoi(prediction);

                    if (number >= 0 && number <= 9)
                    {
                        if (client.sendPrediction(number))
                        {
                            std::cout << "Sent prediction: "
                                      << number << "\n";

                            lastPrediction = prediction;
                        }
                    }
                }
            }

            file.close();
        }

        std::this_thread::sleep_for(
            std::chrono::milliseconds(200)
        );
    }

    return 0;
}