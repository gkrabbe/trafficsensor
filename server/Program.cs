using Microsoft.AspNet.SignalR;
using Microsoft.Extensions.Logging;
using Microsoft.Owin.Hosting;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace server
{
    class Program
    {
        public static ILoggerFactory logger;
        static string url = "http://127.0.0.1:8088/";
        static void Main(string[] args)
        {
            var loggerFactory = LoggerFactory.Create(builder =>
            {
                builder
                    .AddFilter("Microsoft", LogLevel.Warning)
                    .AddFilter("System", LogLevel.Warning)
                    .AddFilter("LoggingConsoleApp.Program", LogLevel.Debug)
                    .AddConsole();
            });

            logger = loggerFactory.AddFile("logs.log");
            var _logger = logger.CreateLogger("Principal");
            var SignalR = WebApp.Start(url);
            _logger.LogInformation($"Server start in {url}");
            while (true)
            {
                _logger.LogInformation($"Press ESC to close aplication");
                if (ConsoleKey.Escape == Console.ReadKey().Key)
                    break;
            }
            _logger.LogInformation($"Server Stopped");
        }
    }

    public class Step5 : Hub
    {
        readonly ILogger loggerC = Program.logger.CreateLogger("ClientStep5");
        readonly ILogger loggerS = Program.logger.CreateLogger("HubStep5");

        static String status = "<SEM INFO>";

        public void SendLog(string msg)
        {
            loggerC.LogInformation($"{msg}");
        }

        public void CountCar(string via,int total)
        {
            loggerC.LogInformation($"Via:'{via}' Com total de {total} automoveis");
        }

        public void Alert(string msg)
        {
            loggerC.LogWarning($"{msg}");
        }

        public void StatusMovimento()
        {
            Clients.Client(Context.ConnectionId).statusMovimento(status);
        }

        public void SetMovimento(string _status)
        {
            loggerC.LogInformation($"Trafego {_status}");
            status = _status;
            Clients.All.statusMovimento(status);
        }

        public override Task OnConnected()
        {
            loggerS.LogWarning($"Client '{Context.ConnectionId}' connected");
            return base.OnConnected();
        }

        public override Task OnDisconnected(Boolean stopCalled)
        {
            loggerS.LogWarning($" Client '{Context.ConnectionId}' disconnected");
            return base.OnDisconnected(stopCalled);
        }

        public override Task OnReconnected()
        {
            loggerS.LogWarning($"Client '{Context.ConnectionId}' reconnected");
            return base.OnReconnected();
        }

    }
}
