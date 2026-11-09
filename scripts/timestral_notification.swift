import AppKit
import UserNotifications

final class Presenter: NSObject, UNUserNotificationCenterDelegate {
    func userNotificationCenter(
        _ center: UNUserNotificationCenter,
        willPresent notification: UNNotification
    ) async -> UNNotificationPresentationOptions {
        [.banner, .list, .sound]
    }
}

let arguments = CommandLine.arguments
guard arguments.count == 4 else { exit(2) }
let mode = arguments[1]
let message = arguments[2]
let receipt = URL(fileURLWithPath: arguments[3])

let app = NSApplication.shared
app.setActivationPolicy(.accessory)
let center = UNUserNotificationCenter.current()
let presenter = Presenter()
center.delegate = presenter

func finish(_ result: [String: Any]) -> Never {
    do {
        try JSONSerialization.data(withJSONObject: result).write(to: receipt, options: .atomic)
    } catch {
        NSLog("Timestral notification receipt failed: %@", String(describing: error))
    }
    exit(result["accepted"] as? Bool == true ? 0 : 1)
}

Task { @MainActor in
    do {
        if mode == "authorize" {
            _ = try await center.requestAuthorization(options: [.alert, .sound])
        }
        let settings = await center.notificationSettings()
        guard settings.authorizationStatus == .authorized, settings.alertSetting == .enabled else {
            finish(["accepted": false, "error": "macOS notifications or alerts are disabled",
                    "authorization_status": settings.authorizationStatus.rawValue])
        }
        let content = UNMutableNotificationContent()
        content.title = "Timestral · Klinehouse"
        content.body = message
        content.sound = .default
        let id = UUID().uuidString
        try await center.add(UNNotificationRequest(identifier: id, content: content, trigger: nil))
        var result: [String: Any] = ["accepted": true, "id": id]
        if mode == "authorize" {
            var delivered = false
            for _ in 0..<12 {
                if await center.deliveredNotifications().contains(where: { $0.request.identifier == id }) {
                    delivered = true
                    break
                }
                try await Task.sleep(for: .milliseconds(250))
            }
            result["delivered"] = delivered
        }
        finish(result)
    } catch {
        finish(["accepted": false, "error": String(describing: error)])
    }
}

app.run()
