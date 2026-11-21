import "./styles.css";
import { mountJobPage } from "./pages/job/job-page";

type Page = (root: HTMLElement, params: URLSearchParams) => void | Promise<void>;

/** Bảng route đơn giản; thêm trang mới (đăng nhập, danh sách job...) ở đây. */
const routes: Record<string, Page> = {
  "/": mountJobPage,
};

const root = document.getElementById("app")!;
const page = routes[location.pathname.replace(/\/+$/, "") || "/"];
if (page) void page(root, new URLSearchParams(location.search));
else root.textContent = "Không tìm thấy trang.";
