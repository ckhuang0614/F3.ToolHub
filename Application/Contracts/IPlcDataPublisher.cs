using System.Threading;
using System.Threading.Tasks;
using F3.ToolHub.Domain.Telemetry;

namespace F3.ToolHub.Application.Contracts;

public interface IPlcDataPublisher
{
    Task PublishAsync(PlcDataSnapshot snapshot, CancellationToken cancellationToken);
}

public interface IPlcDataSink
{
    Task PublishAsync(PlcDataSnapshot snapshot, CancellationToken cancellationToken);
}
